# -*- coding: utf-8 -*-
# @Author : Gan
# @Time : 2024/1/15 10:07

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import torch


def get_optim(model, config):
    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "BatchNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
    else:
        optimizer_parameters = model.parameters()

    if config.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(optimizer_parameters, lr=config.lr, weight_decay=config.weight_decay,
                                    momentum=config.momentum)
    elif config.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(optimizer_parameters, lr=config.lr)
    elif config.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        raise ValueError(f'optimizer {config.optimizer} has not been supported')
    return optimizer


def get_sche(train_dataloader, optimizer, config):
    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end=config.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)

    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)

    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps)

    elif config.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=int(train_steps * 1.05),
                                                    # avoid decrease to 0
                                                    num_warmup_steps=warmup_steps)

    else:
        scheduler = None
        raise ValueError(f'optimizer {config.scheduler} has not been supported')

    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))
    return scheduler
