from models.mlp_mixer import construct_mlp_mixer
from models.resnet import construct_resnet9, construct_resnet34, construct_resnet50
from models.vgg import construct_vgg16

model_list = {
    'mlp_mixer': construct_mlp_mixer,
    'resnet9': construct_resnet9,
    'resnet34': construct_resnet34,
    'resnet50': construct_resnet50,
    'vgg16': construct_vgg16
}
    

