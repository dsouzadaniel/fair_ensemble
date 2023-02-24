from typing import List
import os
import torch as ch
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

ORIG_DATAPATH = './dataset'
FFCV_DATAPATH = './dataset_ffcv'

if not os.path.exists(ORIG_DATAPATH):
    os.makedirs(ORIG_DATAPATH)

if not os.path.exists(FFCV_DATAPATH):
    os.makedirs(FFCV_DATAPATH)


datasets = {
    'train': torchvision.datasets.CIFAR100(ORIG_DATAPATH, train=True, download=True),
    'test': torchvision.datasets.CIFAR100(ORIG_DATAPATH, train=False, download=True)
}


for (name, ds) in datasets.items():
    writer = DatasetWriter(f'./{FFCV_DATAPATH}/cifar_{name}.beton', {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(ds)

print("Datasets Created!")