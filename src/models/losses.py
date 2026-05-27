import torch
import torch.nn as nn
import torch.nn.functional as F


class BANLoss(nn.Module):
    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 1.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def forward(
        self,
        cls_pred: torch.Tensor,
        reg_pred: torch.Tensor,
        cls_target: torch.Tensor,
        reg_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        cls_pred:   [B, 2, H, W]  foreground/background logits
        reg_pred:   [B, 4, H, W]  (dl, dt, dr, db) predictions, non-negative
        cls_target: [B, H, W]     1=positive, 0=negative, -1=ignore
        reg_target: [B, 4, H, W]  (dl, dt, dr, db) ground truth
        """
        pos_mask = cls_target == 1
        neg_mask = cls_target == 0

        cls_loss = self._balanced_cls_loss(cls_pred, cls_target, pos_mask, neg_mask)
        reg_loss = self._iou_loss(reg_pred, reg_target, pos_mask)

        return self.cls_weight * cls_loss + self.reg_weight * reg_loss

    def _balanced_cls_loss(self, pred, target, pos_mask, neg_mask):
        n_pos = pos_mask.sum().clamp(min=1).float()
        n_neg = neg_mask.sum().clamp(min=1).float()

        sample_weight = torch.zeros_like(target, dtype=torch.float)
        sample_weight[pos_mask] = 0.5 / n_pos
        sample_weight[neg_mask] = 0.5 / n_neg

        valid = pos_mask | neg_mask

        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, 2)
        target_flat = target.reshape(-1).long()
        weight_flat = sample_weight.reshape(-1)
        valid_flat = valid.reshape(-1)

        per_sample_loss = F.cross_entropy(
            pred_flat[valid_flat], target_flat[valid_flat], reduction='none'
        )
        return (per_sample_loss * weight_flat[valid_flat]).sum()

    def _iou_loss(self, pred, target, pos_mask):
        if pos_mask.sum() == 0:
            return pred.sum() * 0.0

        # [N, 4] — only at positive locations
        pred_pos = pred.permute(0, 2, 3, 1)[pos_mask]
        target_pos = target.permute(0, 2, 3, 1)[pos_mask]

        dl_p, dt_p, dr_p, db_p = pred_pos.unbind(dim=1)
        dl_t, dt_t, dr_t, db_t = target_pos.unbind(dim=1)

        # Both boxes share the same center, so intersection sides simplify to mins
        inter_w = torch.min(dl_p, dl_t) + torch.min(dr_p, dr_t)
        inter_h = torch.min(dt_p, dt_t) + torch.min(db_p, db_t)
        inter_area = (inter_w * inter_h).clamp(min=0)

        pred_area = (dl_p + dr_p) * (dt_p + db_p)
        target_area = (dl_t + dr_t) * (dt_t + db_t)
        union_area = (pred_area + target_area - inter_area).clamp(min=1e-6)

        iou = inter_area / union_area
        return (1.0 - iou).mean()


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
