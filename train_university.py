import os
import time
import shutil
import sys
import torch
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from utils.util import setup_system, Logger
from utils.optim_sche import get_optim, get_sche
from datasets.university import U1652DatasetEval, U1652DatasetTrain, get_transforms
from utils.trainer import evaluate, train
from utils.losses import get_loss_function
from models.model import GeoModel
import datetime


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

    # Training 
    mixed_precision: bool = True
    custom_sampling: bool = True  # use custom sampling instead of random
    seed = 1
    epochs: int = 40
    batch_size: int = 32  # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0)  # GPU ids for training

    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 1  # eval every n Epoch
    normalize_features: bool = True
    eval_gallery_n: int = -1  # -1 for all or int

    # Optimizer 
    optimizer: str = 'adamw'  # sgd adam adamw
    clip_grad = 100.  # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False  # Gradient Checkpointing

    weight_decay: float = 0.001  # Only for SGD optimizer
    momentum: float = 0.9

    # Loss
    loss: str = 'SemiTriplet'  # SemiTriplet, InfoNCE
    alpha: int = 10

    # Learning Rate
    lr: float = 0.0001  # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "linear"  # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 0
    lr_end: float = 0.0001  # only for "polynomial"

    # Dataset
    dataset: str = 'U1652-D2S'  # 'U1652-D2S' | 'U1652-S2D'
    data_folder: str = "/software/Data/U1652"

    # Augment Images
    prob_flip: float = 0.5  # flipping the sat image and drone image simultaneously
    # Savepath for model checkpoints
    model_path: str = "./checkpoints"
    # Eval before training
    zero_shot: bool = False
    # Checkpoint to start from
    checkpoint_start = None

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 14
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = True
    # make cudnn deterministic
    cudnn_deterministic: bool = False


# -----------------------------------------------------------------------------#
# Train Config                                                                #
# -----------------------------------------------------------------------------#

config = Configuration()
config.query_folder_train = '/software/Data/U1652/train/satellite'
config.gallery_folder_train = '/software/Data/U1652/train/drone'

if config.dataset == 'U1652-D2S':
    config.query_folder_test = '/software/Data/U1652/test/query_drone'
    config.gallery_folder_test = '/software/Data/U1652/test/gallery_satellite'
elif config.dataset == 'U1652-S2D':
    config.query_folder_test = '/software/Data/U1652/test/query_satellite'
    config.gallery_folder_test = '/software/Data/U1652/test/gallery_drone'

if __name__ == '__main__':

    model_path = "{}/{}_{}_{}_{}/{}_lr-{}_loss-{}".format(config.model_path, config.dataset, config.backbone,
                                                          config.attention, config.aggregation,
                                                          time.strftime('%m-%d-%H-%M'), config.lr, config.loss)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train_u1652.txt".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))
    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------

    model = GeoModel(config)
    img_size = (config.img_size, config.img_size)

    # Activate gradient checkpointing
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)
    # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Model to device   
    model = model.to(config.device)

    # -----------------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------------

    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size)

    # Train
    train_dataset = U1652DatasetTrain(query_folder=config.query_folder_train,
                                      gallery_folder=config.gallery_folder_train,
                                      transforms_query=train_sat_transforms,
                                      transforms_gallery=train_drone_transforms,
                                      prob_flip=config.prob_flip,
                                      shuffle_batch_size=config.batch_size,
                                      )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=not config.custom_sampling,
                                  pin_memory=True)

    # Reference Satellite Images
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                          mode="query",
                                          transforms=val_transforms,
                                          )

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    # Query Ground Images Test
    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test,
                                            mode="gallery",
                                            transforms=val_transforms,
                                            sample_ids=query_dataset_test.get_sample_ids(),
                                            gallery_n=config.eval_gallery_n,
                                            )

    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                         batch_size=config.batch_size_eval,
                                         num_workers=config.num_workers,
                                         shuffle=False,
                                         pin_memory=True)

    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))

    # -----------------------------------------------------------------------------#
    # Loss                                                                        #
    # -----------------------------------------------------------------------------#

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2. ** 10)
    else:
        scaler = None

    # -----------------------------------------------------------------------------#
    # optimizer                                                                   #
    # -----------------------------------------------------------------------------#

    optimizer = get_optim(model, config)
    scheduler = get_sche(train_dataloader, optimizer, config)
    loss_function = get_loss_function(config)

    # print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    # print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))

    # -----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    # -----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30 * "-", "Zero Shot", 30 * "-"))

        r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)

    # -----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    # -----------------------------------------------------------------------------#
    if config.custom_sampling:
        train_dataloader.dataset.shuffle()

    # -----------------------------------------------------------------------------#
    # Train                                                                       #
    # -----------------------------------------------------------------------------#
    start_epoch = 0
    best_score = 0

    for epoch in range(1, config.epochs + 1):

        print("\n{}[Epoch: {}]{}".format(30 * "-", epoch, 30 * "-"))
        print(datetime.datetime.now())

        train_loss = train(config,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)

        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))

        # evaluate
        print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))
        r1_test = evaluate(config=config,
                           model=model,
                           query_loader=query_dataloader_test,
                           gallery_loader=gallery_dataloader_test,
                           ranks=[1, 5, 10],
                           step_size=1000,
                           cleanup=True)

        if r1_test > best_score:
            torch.save(model.state_dict(), '{}/best_score.pth'.format(model_path, epoch, r1_test))
            best_score = r1_test

        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
            if r1_test == best_score:
                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(),
                               '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1_test))

        if config.custom_sampling:
            train_dataloader.dataset.shuffle()

    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))
