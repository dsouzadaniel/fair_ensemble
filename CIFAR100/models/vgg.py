import torch as ch

from torchvision.models import vgg16_bn

device = ch.device("cuda" if ch.cuda.is_available() else "cpu")

def construct_vgg16(num_classes):
    model = vgg16_bn(num_classes=num_classes)
    model = model.to(memory_format=ch.channels_last).to(device)
    return model
