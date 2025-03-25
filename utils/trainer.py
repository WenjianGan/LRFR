import torch
import time
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F
import numpy as np
import gc
from utils.util import AverageMeter


def train(config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):
    # set model train mode
    model.train()

    losses = AverageMeter()

    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)

    step = 1

    if config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    # for loop over one epoch
    for query, reference, ids in bar:

        if scaler:
            with autocast():

                # data (batches) to device
                query = query.to(config.device)
                reference = reference.to(config.device)

                # Forward pass
                features1, features2 = model(query, reference)
                loss = loss_function(features1, features2)
                losses.update(loss.item())

            scaler.scale(loss).backward()

            # Gradient clipping
            if config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)

                # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if scheduler is not None:
                scheduler.step()

        else:

            # data (batches) to device
            query = query.to(config.device)
            reference = reference.to(config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            loss = loss_function(features1, features2)
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()

            # Gradient clipping
            if config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)

                # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()

            # Scheduler
            if scheduler is not None:
                scheduler.step()

        if config.verbose:
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr": "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if config.verbose:
        bar.close()

    return losses.avg


def predict(config, model, dataloader):
    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    if config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    img_features_list = []

    ids_list = []
    with torch.no_grad():

        for img, ids in bar:

            ids_list.append(ids)

            with autocast():

                img = img.to(config.device)
                img_feature = model(img)

                # normalize is calculated in fp32
                if config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)

            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))

        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(config.device)

    if config.verbose:
        bar.close()

    return img_features, ids_list


def evaluate(config, model, query_loader, gallery_loader, ranks=[1, 5, 10], step_size=1000, cleanup=True):
    print("Extract Features:")
    img_features_query, ids_query = predict(config, model, query_loader)
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader)

    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()

    print("Compute Scores:")

    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0
    for i in tqdm(range(len(ids_query))):
        ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i], img_features_gallery, gl)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    AP = ap / len(ids_query) * 100

    CMC = CMC.float()
    CMC = CMC / len(ids_query)  # average CMC

    # top 1%
    top1 = round(len(ids_gallery) * 0.01)

    string = []

    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i - 1] * 100))

    string.append('Recall@top1: {:.4f}'.format(CMC[top1] * 100))
    string.append('AP: {:.4f}'.format(AP))

    print(' - '.join(string))

    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()

    return CMC[0]


def eval_query(qf, ql, gf, gl):
    score = gf @ qf.unsqueeze(-1)

    score = score.squeeze().cpu().numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl == -1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc
