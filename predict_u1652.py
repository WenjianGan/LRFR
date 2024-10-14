import os
from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch

from models.model import GeoModel
from utils.trainer import evaluate
from datasets.university import U1652DatasetEval, get_transforms


@dataclass
class Configuration:
    # Model
    backbone: str = 'ConvNeXt'
    attention: str = 'GSRA'
    aggregation: str = 'SALAD'

    num_channels: int = 384
    img_size: int = 384
    num_clusters: int = 128
    cluster_dim: int = 64

    seed = 1
    verbose: bool = True

    # Eval
    batch_size_eval: int = 32
    eval_gallery_n: int = -1  # -1 for all or int
    normalize_features: bool = True

    dataset: str = 'U1652-S2D'  # 'U1652-D2S' | 'U1652-S2D'
    data_folder: str = r"/data/U1652"

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 12
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = True
    # make cudnn deterministic
    cudnn_deterministic: bool = False
    model_path = './checkpoints/best_score.pth'


# -----------------------------------------------------------------------------#
# Config                                                                      #
# -----------------------------------------------------------------------------#

config = Configuration()

if config.dataset == 'U1652-D2S':
    config.query_folder_test = r'/data/U1652/test/query_drone'
    config.gallery_folder_test = r'/data/U1652/test/gallery_satellite'
elif config.dataset == 'U1652-S2D':
    config.query_folder_test = r'/data/U1652/test/query_satellite'
    config.gallery_folder_test = r'/data/U1652/test/gallery_drone'

if __name__ == '__main__':
    val_transforms = get_transforms((config.img_size, config.img_size))
    model = GeoModel(config)
    model.load_state_dict(torch.load(config.model_path))
    model = model.to(config.device)

    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test, mode="query",
                                          transforms=val_transforms)

    query_dataloader_test = DataLoader(query_dataset_test, batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers, shuffle=False, pin_memory=True)

    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test, mode="gallery",
                                            transforms=val_transforms, sample_ids=query_dataset_test.get_sample_ids(),
                                            gallery_n=config.eval_gallery_n, )

    gallery_dataloader_test = DataLoader(gallery_dataset_test, batch_size=config.batch_size_eval,
                                         num_workers=config.num_workers, shuffle=False, pin_memory=True)

    r1_test = evaluate(config=config,
                       model=model,
                       query_loader=query_dataloader_test,
                       gallery_loader=gallery_dataloader_test,
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)
