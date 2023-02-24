import numpy as np
import torch

from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
    """
    From https://github.com/weiaicunzai/pytorch-cifar100/blob/11d8418f415b261e4ae3cb1ffe20d06ec95b98e4/utils.py#L234
    warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def rand_bbox(size, lam, gen):
    """
    Based on https://github.com/omihub777/MLP-Mixer-CIFAR/blob/main/utils.py#L25
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = torch.randint(W,(1,),generator=gen).item()
    cy = torch.randint(H,(1,),generator=gen).item()

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CutOut(torch.nn.Module):
    """
    """
    def __init__(self, crop_size, fill):
        self.crop_size = crop_size
        self.fill = np.array(fill)

    def __call__(self, images):
        crop_size = self.crop_size
        fill = torch.cuda.ByteTensor(np.zeros((3,crop_size,crop_size)))
        fill[0,:,:] = self.fill[0]
        fill[1,:,:] = self.fill[1]
        fill[2,:,:] = self.fill[2]

        for i in range(images.shape[0]):
            # Generate random origin
            coord = (
                torch.randint(images.shape[2] - crop_size + 1,(1,)).item(),
                torch.randint(images.shape[3] - crop_size + 1,(1,)).item(),
            )
            # Black out image in-place
            images[i, :, coord[0]:coord[0] + crop_size, coord[1]:coord[1] + crop_size] = fill

        return images