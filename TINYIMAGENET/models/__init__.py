from models.resnet import (
    construct_resnet9, 
    construct_resnet18,
    construct_resnet34, 
    construct_resnet50
    )
from models.vgg import construct_vgg16
from models.vit import construct_vit 

model_list = {
    'resnet9': construct_resnet9,
    'resnet18': construct_resnet18,
    'resnet34': construct_resnet34,
    'resnet50': construct_resnet50,
    'vgg16': construct_vgg16, 
    'vit': construct_vit,
}
    

