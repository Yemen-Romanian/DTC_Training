import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedLoss(nn.Module):
    def __init__(self, pos_threshold=0.1):
        super(BalancedLoss, self).__init__()
        self.pos_threshold = pos_threshold

    def forward(self, prediction, target):
        """
        prediction: [B, 17, 17] - network's logits
        target:     [B, 17, 17] - gt mask (binary or gaussian)
        """
        pos_mask = (target >= self.pos_threshold).float()
        neg_mask = (target <  self.pos_threshold).float()

        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()

        weight = torch.zeros_like(target)
        if n_pos > 0:
            weight += pos_mask * (0.5 / n_pos)
        if n_neg > 0:
            weight += neg_mask * (0.5 / n_neg)

        loss = F.binary_cross_entropy_with_logits(
            prediction, target, weight=weight, reduction='sum'
        )

        return loss / prediction.size(0)
