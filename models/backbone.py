import torch
import torchvision
from torch import nn


class ConvNextTiny(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        layers = list(model.features.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.backbone(x)  # (B, 768, H/16, W/16)


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        layers = list(model.children())[:-2]
        self.backbone = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.backbone(x)  # (B, 512, H/32, W/32)


def get_backbone(backbone='ConvNeXt'):
    if backbone.lower() == 'convnext':
        return ConvNextTiny()
    else:
        return ResNet18()
