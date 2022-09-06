import torch
from torch import nn
import torch.nn.functional as F
from fcos_core.modeling.utils import bbox_overlap

# distributionfocalloss
class DistributionFocalLoss(nn.Module):
    def __init__(self, avg_factor=8.0):
        super(DistributionFocalLoss, self).__init__()
        self.avg_factor = avg_factor
    
    def forward(self, box_regression, reg_targets, dfl_weights=None):
        """
        box_regression:   N, 8*(16+1)
        reg_targets:      N, 8
        """
        reg_targets_ = (torch.clamp(reg_targets, -8, 8) + 8).reshape(-1)
        dist_l = reg_targets_.long()
        dist_r = dist_l + 1

        wl = dist_r - reg_targets_
        wr = reg_targets_ - dist_l

        box_regression = box_regression.reshape(-1, 8, 16+1).reshape(-1, 16+1)
        loss = F.cross_entropy(box_regression, dist_l, reduction='none') * wl + F.cross_entropy(box_regression, dist_r, reduction='none') * wr
        if dfl_weights is not None:
            loss = loss * dfl_weights
        return loss.mean()

# qualityfocalloss
class SigmoidQualityFocalLoss(nn.Module):
    def __init__(self, beta=2.0):
        super(SigmoidQualityFocalLoss, self).__init__()
        self.beta = beta
        self.func = F.binary_cross_entropy_with_logits

    def forward(self, cls_logits, cls_targets, box_regression, reg_targets, reg_anchors):
        # negatives are supervised by 0 quality score
        cls_sigmoid = cls_logits.sigmoid()
        scale_factor = cls_sigmoid
        zerolabel = scale_factor.new_zeros(cls_logits.shape)
        loss = self.func(cls_logits, zerolabel, reduction='none') * scale_factor.pow(self.beta)

        # positives are supervised by bbox quality (IoU) score
        pos = (cls_targets > 0).nonzero().squeeze(1)
        pos_label = cls_targets[pos].long() - 1
        
        # calculate score
        score = cls_logits.new_zeros(cls_targets.size())
        reg_anchors = reg_anchors[pos].repeat(1, box_regression.shape[-1]//reg_anchors.shape[-1])
        box_preds = reg_anchors - box_regression[pos]
        box_targets = reg_anchors - reg_targets[pos]
        score[pos] = bbox_overlap(box_preds, box_targets, is_aligned=True)

        scale_factor = score[pos] - cls_sigmoid[pos, pos_label]
        loss[pos, pos_label] = self.func(cls_logits[pos, pos_label], score[pos], reduction='none') * scale_factor.abs().pow(self.beta)
        if torch.isnan(loss[pos, pos_label]).any():
            import ipdb; ipdb.set_trace()
        loss = loss.sum()
        return loss

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "beta=" + str(self.beta)
        tmpstr += ")"
        return tmpstr