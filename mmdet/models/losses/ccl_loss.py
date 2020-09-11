import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weighted_loss
from ..registry import LOSSES


@weighted_loss
def ccl_loss(prediction,
            target, 
            eps=1e-6):

    # indexing and loss computation
    
    pair = target['pair']


    target1 = target['target'][pair[:, 0], pair[:, 1]]
    target2 = target['target'][pair[:, 2], pair[:, 3]]
    sm1 = prediction[pair[:, 0], :, pair[:, 1]]
    sm2 = prediction[pair[:, 2], :, pair[:, 3]]


    sm1 /= sm1.sum(-1, keepdim=True) + eps
    sm2 /= sm2.sum(-1, keepdim=True) + eps

    R = torch.eq(target1, target2).float()
    
    '''
    conf1 = target['score'][pair[:, 0], pair[:, 1]]
    conf2 = target['score'][pair[:, 2], pair[:, 3]]

    R_conf = conf1 * conf2
    '''
    P = (sm1 * sm2).sum(-1)

    #L = R_conf * (R * P + (1 - R) * (1 - P)) + eps
    L = (R * P + (1 - R) * (1 - P)) + eps
    L = -L.log()      
    return L


@LOSSES.register_module
class ConstrainedClusteringLikelihoodLoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(ConstrainedClusteringLikelihoodLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                score,
                pair,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs): 
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        target=dict(target=target, score=score, pair=pair)

        loss = self.loss_weight * ccl_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

def _apply_margin(x, m):
    if isinstance(m, float):
        return (x + m).clamp(min=0)
    elif m.lower() == "soft":
        return F.softplus(x)
    elif m.lower() == "none":
        return x
    else:
        raise NotImplementedError("The margin %s is not implemented in BatchHard!" % m)

def calc_cdist(a, b, metric='euclidean'):
    diff = a[:, None, :] - b[None, :, :]
    if metric == 'euclidean':
        return torch.sqrt(torch.sum(diff*diff, dim=2) + 1e-12)
    elif metric == 'sqeuclidean':
        return torch.sum(diff*diff, dim=2)
    elif metric == 'cityblock':
        return torch.sum(diff.abs(), dim=2)
    else:
        raise NotImplementedError("Metric %s has not been implemented!" % metric)

@weighted_loss
def batch_hard(cdist, pids, margin):
    """Computes the batch hard loss as in arxiv.org/abs/1703.07737.

    Args:
        cdist (2D Tensor): All-to-all distance matrix, sized (B,B).
        pids (1D tensor): PIDs (classes) of the identities, sized (B,).
        margin: The margin to use, can be 'soft', 'none', or a number.
    """
    mask_pos = (pids[None, :] == pids[:, None]).float()

    ALMOST_INF = 9999.9
    furthest_positive = torch.max(cdist * mask_pos, dim=0)[0]
    furthest_negative = torch.min(cdist + ALMOST_INF*mask_pos, dim=0)[0]
    #furthest_negative = torch.stack([
    #    torch.min(row_d[row_m]) for row_d, row_m in zip(cdist, mask_neg)
    #]).squeeze() # stacking adds an extra dimension

    return _apply_margin(furthest_positive - furthest_negative, margin)

@LOSSES.register_module
class BatchHard(nn.Module):
    def __init__(self, m, reduction='mean', loss_weight=1.0):
        super(BatchHard, self).__init__()
        self.name = "BatchHard(m={})".format(m)
        self.m = m
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                pids,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        dist = calc_cdist(pred, pred)
        batch_hard_loss = self.batch_hard(  dist,
                                            pids, 
                                            self.m,
                                            reduction=reduction,
                                            avg_factor=avg_factor,
                                            **kwargs)

        return batch_hard_loss


        

