import torch as ch

from torchvision.models import resnet34, resnet50

device = ch.device("cuda" if ch.cuda.is_available() else "cpu")

class Mul(ch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x): return x * self.weight


class Flatten(ch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


class Residual(ch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x): return x + self.module(x)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return ch.nn.Sequential(
        ch.nn.Conv2d(channels_in, channels_out,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     groups=groups, bias=False),
        ch.nn.BatchNorm2d(channels_out),
        ch.nn.ReLU(inplace=True)
    )

def construct_resnet9(num_classes):
    model = ch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(ch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        ch.nn.MaxPool2d(2),
        Residual(ch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        ch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        ch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=ch.channels_last).to(device)
    # print(f"Current Model Init Weights Sum : {np.sum([p.detach().cpu().numpy().sum() for p in model.parameters()])}")

    return model

def construct_resnet34(num_classes):
    model = resnet34(num_classes=num_classes)
    model = model.to(memory_format=ch.channels_last).to(device)
    return model

def construct_resnet50(num_classes):
    model = resnet50(num_classes=num_classes)
    model = model.to(memory_format=ch.channels_last).to(device)
    return model
