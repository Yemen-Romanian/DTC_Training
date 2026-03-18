import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedLoss(nn.Module):
    def __init__(self):
        super(BalancedLoss, self).__init__()

    def forward(self, prediction, target):
        """
        prediction: [B, 1, 17, 17] - network's logits
        target: [B, 1, 17, 17] - binary gt mask
        """
        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()

        weight = torch.zeros_like(target)
        if n_pos > 0:
            weight[target == 1] = 0.5 / n_pos
        if n_neg > 0:
            weight[target == 0] = 0.5 / n_neg

        loss = F.binary_cross_entropy_with_logits(
            prediction, target, weight=weight, reduction='sum'
        )

        return loss / prediction.size(0)
