import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_backbone
from .attention import get_attention
from .aggregation import get_aggregation


class LRFR(nn.Module):
    def __init__(self, config):
        super(LRFR, self).__init__()
        self.backbone = get_backbone(backbone=config.backbone)
        self.attention = get_attention(attention=config.attention, channel=config.num_channels,
                                       spatial=((config.img_size // 16) * (config.img_size // 16)))
        self.aggregation = get_aggregation(aggregation=config.aggregation, num_channels=config.num_channels,
                                           num_clusters=config.num_clusters, cluster_dim=config.cluster_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        x = self.aggregation(x)
        return F.normalize(x.flatten(1), p=2, dim=1)


class GeoModel(nn.Module):
    def __init__(self, config):
        super(GeoModel, self).__init__()
        self.model = LRFR(config=config)

    def forward(self, img1, img2=None):

        if img2 is not None:
            image_features1 = self.model(img1)
            image_features2 = self.model(img2)
            return image_features1, image_features2

        else:
            image_features = self.model(img1)
            return image_features

