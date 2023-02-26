from vit_pytorch import ViT
import torch as ch

device = ch.device("cuda" if ch.cuda.is_available() else "cpu")


def construct_vit(num_classes):
    model = ViT(
        image_size=64,
        patch_size=8,
        num_classes=num_classes,
        dim=192,
        depth=9,
        heads=12,
        mlp_dim=192 * 2,
        dropout=0,
        emb_dropout=0
    )
    model = model.to(memory_format=ch.channels_last).to(device)
    return model
