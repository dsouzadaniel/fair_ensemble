"""
Fast training script for TINY-IMAGENET using FFCV.

Then, simply run this to train models with default hyperparameters:

    `python train_tinyimagenet.py --config-file default_config.yaml`

You can override arguments as follows:

    `python train_tinyimagenet.py --config-file default_config.yaml \
                           --training.lr 0.2 --training.num_workers 4 ... [etc]`

or by using a different config file.
"""

# Libraries
import os
import random
from utils import CosineAnnealingWarmupRestarts, build_scheduler, cutmix_data, mixup_criterion, mixup_data
import numpy as np
from numba import njit
@njit
def numba_set_seed(value):
    np.random.seed(value)
import torch as ch

from argparse import ArgumentParser
import time
import datetime
import json
from typing import List
import enlighten
import torchvision
from uuid import uuid4
from pathlib import Path

from const import TINYIMAGENET_CLASSES

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder,CenterCropRGBImageDecoder
# from ffcv.loader import Loader, OrderOption
from ffcv.loader import OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

from ffcv_ext.loader import Loader
from ffcv_ext.ops import RandomHorizontalFlip
from ffcv_ext.resized_crop import RandomResizedCropDecoder

from models import model_list

from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, lr_scheduler, AdamW

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf

from warmup_scheduler import GradualWarmupScheduler


Section('training', 'Hyperparameters').params(
    optim_name=Param(str, 'optimizer to use', default="adam"),
    scheduler_name=Param(str, 'scheduler to use', default="default"),
    da_recipe=Param(str, 'Which data augmentation recipe to use', default="default"),
    lr=Param(float, 'The learning rate to use', required=True),
    epochs=Param(int, 'Number of epochs to run for', required=True),
    lr_peak_epoch=Param(int, 'Peak epoch for cyclic lr', default=5),
    min_lr=Param(float, 'min lr for cosine scheduler', default=1e-6),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.1),
    num_workers=Param(int, 'The number of workers', default=8)
)

Section('eval', 'Hyperparameters').params(
    lr_tta=Param(bool, 'Test Time augmentation by averaging with horizontally flipped version', default=True)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.beton file to use for training', required=True),
    eval_dataset=Param(str, '.beton file to use for validation', required=True),
)

Section('logging', 'how to log stuff').params(
    log_level=Param(int, '0 if only at end 1 otherwise', default=1)
)

Section('seed', 'to make runs reproducible').params(
    default=Param(int, 'default seed', default=789),
    model_init=Param(int, 'model init seed', default=789),
    batch_order=Param(int, 'batch order seed', default=789),
    DA=Param(int, 'data augmentation seed', default=789),
)

Section('exp', 'experiment files related').params(
    folder=Param(str, 'output folder', default='temp'),
    ablation=Param(str, 'current ablation', default='temp'),
    ix=Param(int, 'current run index', default=0),
)

Section('model', 'model to use').params(
    model_name=Param(str, And(str, OneOf(model_list)), default='resnet9'),
)


manager = enlighten.get_manager()

@param('seed.model_init')
@param('seed.batch_order')
@param('exp.folder')
@param('exp.ablation')
@param('exp.ix')
def prep_folder(folder, ablation, ix, model_init, batch_order):
    _output_folder = os.path.join(folder, ablation, f"ix_{ix}_MS_{model_init}_BS_{batch_order}")
    folder = Path(_output_folder).absolute()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


@param('seed.default')
@param('seed.model_init')
def set_seeds(default, model_init):
    ######### Setting Seeds for Reproducibility #########
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ["PYTHONHASHSEED"] = str(default)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(default)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(default)
    # 4. Set `numba` pseudo-random generator at a fixed value
    numba_set_seed(default)
    # 5. Set Torch seed at a fixed value
    ch.manual_seed(model_init)
    ch.backends.cudnn.benchmark = False
    ch.backends.cudnn.deterministic = True
    ####################################################
    return


@param('data.train_dataset')
@param('data.eval_dataset')
@param('training.da_recipe')
@param('seed.batch_order')
@param('seed.DA')
@param('model.model_name')
@param('training.batch_size')
@param('training.num_workers')
def make_dataloaders(
    train_dataset, 
    model_name, 
    da_recipe, 
    eval_dataset, 
    batch_order, 
    DA, 
    batch_size, 
    num_workers
):
    paths = {
        'train': train_dataset,
        'eval': eval_dataset

    }
    loaders = {}

    dataset_mean = np.array([0.485, 0.456, 0.406])*255
    dataset_std = np.array([0.229, 0.224, 0.225])*255 

    if da_recipe == "default":
        da_recipe = model_name
    else:
       da_recipe = da_recipe 

    DA_regiments = {
        'vit': [
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(64, padding=4), 
            torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.IMAGENET),
            Convert(ch.float16),
            torchvision.transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.4), ratio=(0.3, 1/0.3),
                value=dataset_mean.tolist()
            ),
            torchvision.transforms.Normalize(dataset_mean, dataset_std),
        ], 
        'resnet50': [
            RandomHorizontalFlip(seed=DA),
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(dataset_mean, dataset_std),
        ], 
        'resnet34': [
            RandomHorizontalFlip(seed=DA),
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(dataset_mean, dataset_std),
        ],
        'resnet9': [
            RandomHorizontalFlip(seed=DA),
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(dataset_mean, dataset_std),
        ], 
        'resnet18': [
            RandomHorizontalFlip(seed=DA),
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(dataset_mean, dataset_std),
        ], 
        'vgg16': [
            RandomHorizontalFlip(seed=DA),
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(dataset_mean, dataset_std),
        ],
    } 

    train_DA = DA_regiments[da_recipe]

    for name in ['train', 'eval']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        image_pipeline[0] = CenterCropRGBImageDecoder((64, 64), 1)
        if name == 'train':
            if da_recipe != "vit": 
                image_pipeline[0] = RandomResizedCropDecoder((64,64), seed=DA)
                # image_pipeline[0] = SimpleRGBImageDecoder()
            image_pipeline.extend(train_DA)
        else: 
            image_pipeline.extend([
                ToTensor(),
                ToDevice('cuda:0', non_blocking=True),
                ToTorchImage(),
                Convert(ch.float16),
                torchvision.transforms.Normalize(dataset_mean, dataset_std),
            ])
        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name], seed=batch_order, batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders

@param('model.model_name')
def construct_model(model_name):
    return model_list[model_name](TINYIMAGENET_CLASSES)

class TinyImagenetTrainer:
    @param('model.model_name')
    @param('seed.DA')
    def __init__(self, model_name, DA, output_folder='logs'):
        self.all_params = get_current_config()

        self.uid = str(uuid4())

        self.initialize_logger(folder=output_folder)

        self.model = construct_model()
        print(f"Current Model Init Weights Sum : {np.sum([p.detach().cpu().numpy().sum() for p in self.model.parameters()])}")
        self.model_name = model_name
        #print(self.model)
        # set data augmentation seed
        ch.manual_seed(DA)

        self.create_optimizer() 

        self.g_cuda = ch.Generator(device='cuda')
        self.g_cpu = ch.Generator()

    def init_weights(self, m):
        if isinstance(m, ch.nn.Linear):
            ch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)
        elif isinstance(m, ch.nn.Conv2d):
            ch.nn.init.xavier_uniform_(m.weight)
    
    @param('training.optim_name')
    @param('training.lr')
    @param('training.momentum')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    def create_optimizer(self, optim_name, lr, momentum, weight_decay, label_smoothing):

        optim_dict = {
            "sgd": SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
            "adam": Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay),
            "adamw": AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
        }
        
        optim = optim_dict[optim_name]

        self.optimizer = optim
        self.loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    @param('training.epochs')
    @param('training.lr_peak_epoch')
    @param('training.scheduler_name')
    @param('training.lr')
    @param('training.min_lr')
    @param('logging.log_level')
    def train(
        self, 
        loaders,
        output_folder, 
        epochs,
        scheduler_name, 
        lr_peak_epoch,
        lr,
        min_lr,
        log_level
        ):
        self.scheduler_name = scheduler_name

        if self.scheduler_name == 'default':
            # Cyclic LR with single triangle
            lr_schedule = np.interp(np.arange(epochs + 1),
                                    [0, lr_peak_epoch, epochs],
                                    [0, 1,0])
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_schedule.__getitem__)
        elif self.scheduler_name == 'vit_default':
            self.base_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,  T_max=epochs, eta_min=min_lr)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=10, after_scheduler=self.base_scheduler)
        else:
            raise NotImplementedError

        #self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones= [50,70], gamma=0.1)
        #self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.9, patience=2)
        self.scaler = GradScaler()

        class_accuracy_collect = {"train": [], "eval": []}

        progress_bar = manager.counter(total=epochs, desc="Epochs", unit="epochs", color="red")
        self.curr_lr = self.optimizer.param_groups[0]['lr']
        for epoch in range(epochs):
            train_metrics = self.train_loop(loaders['train'])
            eval_metrics = self.eval_loop(loaders['eval'])

            # Log Train/Eval Metrics
            if log_level > 0:
                self.log(dict(train_metrics, **{'epoch': epoch}))
                self.log(dict(eval_metrics, **{'epoch': epoch}))

            class_accuracy_collect["train"].append(train_metrics["class_accuracy"])
            class_accuracy_collect["eval"].append(eval_metrics["class_accuracy"])
            epoch_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.update()
            
            print(
                f"|Epoch : {epoch}|\t Train_Acc :{train_metrics['accuracy']:.1f}\t Train_Loss: {train_metrics['loss']:.1f}\t Eval_Acc: {eval_metrics['accuracy']:.1f}\t Eval_Loss:{eval_metrics['loss']:.1f}\t")

            if self.curr_lr!=epoch_lr:
                print(f"LR Changed from {self.curr_lr} to {epoch_lr}")
                self.curr_lr = epoch_lr
        # Write NPY files
        for subset in ["train", "eval"]:
            np.save(os.path.join(output_folder, f'{subset}_per_class_acc.npy'),
                    np.stack(class_accuracy_collect[subset], axis=0))

        # Save checkpoint
        ckpt_pth = os.path.join(output_folder, f'model.pt')
        ch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': eval_metrics['loss'],
                }, ckpt_pth)

    @param('training.da_recipe') 
    def train_loop(self, loader, da_recipe):
        start_time = time.time()
        self.model.train()
        total_loss, num_batches = 0., 0.
        all_preds, all_labs = [], []

        #progress_bar = manager.counter(total=len(loader), desc="Train", unit="batches", color="blue")
        for ims, labs in loader:
            self.optimizer.zero_grad(set_to_none=True)
            if da_recipe == 'vit':
                r = random.uniform(0, 1)
                if r < 0.5:
                    switching_prob = random.uniform(0, 1)
                    
                    # Cutmix
                    if switching_prob < 0.5:
                        slicing_idx, y_a, y_b, lam, sliced = cutmix_data(ims, labs, 1, self.g_cuda, self.g_cpu)
                        ims[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
                        
                        with autocast():
                            out = self.model(ims)
                            loss =  mixup_criterion(self.loss_fn, out, y_a, y_b, lam)
                        
                    # Mixup
                    else:
                        ims, y_a, y_b, lam = mixup_data(ims, labs, 1, self.g_cuda)
                        with autocast():
                            out = self.model(ims)
                            loss = mixup_criterion(self.loss_fn, out, y_a, y_b, lam) 
                else: 
                    with autocast():
                        out = self.model(ims)
                        loss = self.loss_fn(out, labs)
            else:
                with autocast():
                    out = self.model(ims)
                    loss = self.loss_fn(out, labs)

            total_loss += loss.item()
            num_batches += 1.0

            preds = out.argmax(1)
            all_preds.extend(preds.detach().cpu())
            all_labs.extend(labs.detach().cpu())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
        self.scheduler.step()

            #progress_bar.update()

        all_preds = np.array(all_preds)
        all_labs = np.array(all_labs)
        # print(f"First Batch First 10 Labels : {all_labs[:10]}")

        # Overall Accuracy
        acc = (all_preds == all_labs).sum() / len(all_preds)

        # Class-level Accuracy
        class_acc = []
        for cix in range(TINYIMAGENET_CLASSES):
            ixs_of_interest = np.where(all_labs == cix)[0]
            class_acc.append(
                round(100 * (all_labs[ixs_of_interest] == all_preds[ixs_of_interest]).sum() / len(ixs_of_interest), 2))

        time_taken = time.time() - start_time
        metrics = {
            'dataset': 'train',
            'loss': total_loss / num_batches,
            'accuracy': acc * 100,
            'class_accuracy': class_acc,
            'time_taken': time_taken
        }
        return metrics

    @param('eval.lr_tta')
    def eval_loop(self, loader, lr_tta):
        start_time = time.time()
        self.model.eval()
        with ch.no_grad():
            total_loss, num_batches = 0., 0.
            all_preds, all_labs = [], []

            #progress_bar = manager.counter(total=len(loader), desc="Eval", unit="batches", color="green")
            for ims, labs in loader:
                with autocast():
                    if lr_tta:
                        out = (self.model(ims) + self.model(ch.fliplr(ims))) / 2.  # Test-time augmentation
                    else:
                        out = self.model(ims)
                    loss = self.loss_fn(out, labs)

                total_loss += loss.item()
                num_batches += 1.0

                preds = out.argmax(1)
                all_preds.extend(preds.detach().cpu())
                all_labs.extend(labs.detach().cpu())

                #progress_bar.update()
        all_preds = np.array(all_preds)
        all_labs = np.array(all_labs)

        # Overall Accuracy
        acc = (all_preds == all_labs).sum() / len(all_preds)

        # Scheduler Update
        #self.scheduler.step(total_loss/num_batches)
        
        
        # Class-level Accuracy
        class_acc = []
        for cix in range(TINYIMAGENET_CLASSES):
            ixs_of_interest = np.where(all_labs == cix)[0]
            class_acc.append(
                round(100 * (all_labs[ixs_of_interest] == all_preds[ixs_of_interest]).sum() / len(ixs_of_interest), 2))

        time_taken = time.time() - start_time
        metrics = {
            'dataset': 'eval',
            'loss': total_loss / num_batches,
            'accuracy': acc * 100,
            'class_accuracy': class_acc,
            'time_taken': time_taken
        }
        return metrics

    def initialize_logger(self, folder):
        folder = (Path(folder) / str(self.uid)).absolute()
        folder.mkdir(parents=True, exist_ok=True)

        self.log_folder = folder
        self.start_time = time.time()

        print(f'=> Logging in {self.log_folder}')
        params = {
            '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
        }

        with open(folder / 'params.json', 'w+') as handle:
            json.dump(params, handle)

    def log(self, content):
        cur_time = time.time()
        with open(self.log_folder / 'log', 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast TINYIMAGENET training')
    config.augment_argparse(parser)
    # Also loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    # Set Seeds to make reproducible runs
    set_seeds()

    # Create Output Folders
    output_folder = prep_folder()

    # Create Dataloaders
    start_time = time.time()
    loaders = make_dataloaders()

    # Train Model
    trainer = TinyImagenetTrainer(output_folder=output_folder)
    trainer.train(loaders=loaders, output_folder=output_folder)

    print(f'Total Time: {datetime.timedelta(seconds=time.time() - start_time)}')
