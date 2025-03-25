import torch
import torch.nn as nn
import torch.distributed.nn


class SoftTripletLoss(nn.Module):
    def __init__(self, margin=None, alpha=10):
        super(SoftTripletLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, inputs_q, inputs_k):
        loss_1 = self.single_forward(inputs_q, inputs_k)
        loss_2 = self.single_forward(inputs_k, inputs_q)
        return (loss_1 + loss_2) * 0.5

    def single_forward(self, inputs_q, inputs_k):
        n = inputs_q.size(0)

        normalized_inputs_q = inputs_q / torch.norm(inputs_q, dim=1, keepdim=True)
        normalized_inputs_k = inputs_k / torch.norm(inputs_k, dim=1, keepdim=True)
        # Compute similarity matrix
        sim_mat = torch.matmul(normalized_inputs_q, normalized_inputs_k.t())

        # split the positive and negative pairs
        eyes_ = torch.eye(n).cuda()

        pos_mask = eyes_.eq(1)
        neg_mask = ~pos_mask

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        pos_sim_ = pos_sim.unsqueeze(dim=1).expand(n, n - 1)
        neg_sim_ = neg_sim.reshape(n, n - 1)

        loss_batch = torch.log(1 + torch.exp((neg_sim_ - pos_sim_) * self.alpha))
        if torch.isnan(loss_batch).any():
            print(inputs_q, inputs_k)
            raise Exception

        loss = loss_batch.mean()
        return loss


class SemiSoftTriHard(nn.Module):
    def __init__(self, alpha, batch_size):
        super().__init__()
        self.alpha = alpha
        self.batch_size = batch_size

    def forward(self, grd_global, sat_global):
        dist_array = 2.0 - 2.0 * torch.matmul(sat_global, grd_global.t())
        pos_dist = torch.diag(dist_array)

        logits = dist_array
        hard_neg_dist_g2s = get_semi_hard_neg(logits, pos_dist)
        hard_neg_dist_s2g = get_semi_hard_neg(logits.t(), pos_dist)
        # hard_neg_dist_g2s = get_topk_hard_neg(logits, pos_dist, int(self.batch_size ** 0.5))
        # hard_neg_dist_s2g = get_topk_hard_neg(logits.t(), pos_dist, int(self.batch_size ** 0.5))
        return (weighted_soft_margin_loss(pos_dist - hard_neg_dist_g2s.t(), self.alpha) + weighted_soft_margin_loss(
            pos_dist - hard_neg_dist_s2g.t(), self.alpha)) / 2.0


def get_topk_hard_neg(logits, pos_dist, k):
    N = logits.shape[0]
    targets = torch.arange(0, N, dtype=torch.long, device=logits.device)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[targets, targets] = False
    hard_neg_dist, _ = torch.topk(logits[mask].reshape(N, -1), largest=False, k=k, dim=1)
    return hard_neg_dist


def get_semi_hard_neg(logits, pos_dist):
    N=logits.shape[0]
    targets = torch.arange(0, N, dtype=torch.long, device=logits.device)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask_=torch.lt(logits, pos_dist.unsqueeze(1))
    mask[targets, targets] = False
    mask_[targets, targets] = False

    mininum=torch.mul(logits, mask_)
    # mininum=torch.masked_select(logits, mask_)

    hard_neg_dist1, _ = torch.max(mininum, 1)
    hard_neg_dist2, _ = torch.min(logits[mask].reshape(N,-1), 1)
    hard_neg_dist = torch.max(hard_neg_dist1, hard_neg_dist2)

    return hard_neg_dist


def weighted_soft_margin_loss(diff, beta=10.0, reduction=torch.mean):
    out = torch.log(1 + torch.exp(diff * beta))
    if reduction:
        out = reduction(out)
    return out


def get_loss_function(config):
    if config.loss.lower() == 'triplet':
        return SoftTripletLoss(alpha=config.alpha)
    elif config.loss.lower() == 'semitriplet':
        return SemiSoftTriHard(alpha=config.alpha, batch_size=config.batch_size)
