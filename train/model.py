import torch
import segmentation_models_pytorch as smp

def build_model() -> torch.nn.Module:
    # Create a U-Net model with a ResNet-34 encoder pre-trained on ImageNet
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = 8,
        activation      = None,
    )

    return model