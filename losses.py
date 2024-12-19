import torch
import torch.nn.functional as F
from typing import Optional, List, Union


class GridLoss(torch.nn.Module):
    """
    Ensure completely correct output grids
    1 if complete grid is solved (prediction == target), 0 otherwise
    """
    def __init__(self, accuracy: str = "absolute"):
        """
        accuracy (str, optional): accuracy measure to use.
                                  "absolute": 1 when all pixels in a grid are correct
                                  "partial": mean across all pixels
        """
        super(GridLoss, self).__init__()
        self.accuracy = accuracy

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)

        if self.accuracy == "absolute":
            grid_score = (pred_labels == targets).all(dim=(1, 2)).float()
        elif self.accuracy == "partial":
            grid_score = (pred_labels == targets).float().mean(dim=(1, 2))
        else:
            raise ValueError(f"Invalid accuracy measure: {self.accuracy}")

        return 1 - grid_score.mean()


class FocalLoss(torch.nn.Module):
    """Introduced in
    'Focal Loss for Dense Object Detection'
    (https://arxiv.org/abs/1708.02002)
    """
    def __init__(self, alpha: Optional[Union[torch.Tensor, float]] = None, gamma: float = 2.0):
        """
        alpha: Tensor of shape (C,) for per-class weights or a scalar for uniform weighting.
        gamma: focusing parameter, Default: 2 (resulted in good results in the original paper)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)  # Log probabilities
        probs = torch.exp(log_probs)  # Probabilities
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        focal_weight = ((1 - probs) ** self.gamma) * targets_one_hot

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_weight = self.alpha.view(1, -1, 1, 1)  # broadcasting
                focal_weight = focal_weight * alpha_weight
            else:  # Scalar case
                focal_weight = focal_weight * self.alpha

        loss = -focal_weight * log_probs
        loss = loss.sum(dim=1)  # Sum across channels (class dimension)

        return loss.mean()


class DiceLoss(torch.nn.Module):
    """
    Introduced in
    Dice Loss: 'V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation'
    (https://arxiv.org/abs/1606.04797)

    Generalized DL: 'Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations'
    (https://arxiv.org/abs/1707.03237)
    """
    def __init__(self, weight: Optional[torch.Tensor] = None, epsilon: float = 1e-6):
        """
        weight: class weights
        epsilon: constant to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        logits = F.softmax(logits, dim=1)

        intersection = torch.sum(logits * targets_one_hot, dim=(2, 3))
        union = torch.sum(logits + targets_one_hot, dim=(2, 3))

        if self.weight is not None:
            intersection = self.weight * intersection
            union = self.weight * union

        dice_scores = (2 * intersection + self.epsilon) / (union + self.epsilon)
        return 1 - dice_scores.mean()


class TverskyLoss(torch.nn.Module):
    """Introduced in
    'Tversky loss function for image segmentation using 3D fully convolutional deep networks'
    (https://arxiv.org/abs/1706.05721)
    """
    def __init__(
        self,
        alpha: float = 0.3, beta: float = 0.7, epsilon: float = 1e-6,
        weight: Optional[torch.Tensor] = None
    ):
        """
        alpha: Class weights
        beta:
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        logits = F.softmax(logits, dim=1)

        true_pos = torch.sum(logits * targets_one_hot, dim=(2, 3))
        false_neg = torch.sum(targets_one_hot * (1 - logits), dim=(2, 3))
        false_pos = torch.sum((1 - targets_one_hot) * logits, dim=(2, 3))

        if self.weight is not None:
            true_pos = self.weight * true_pos
            false_neg = self.weight * false_neg
            false_pos = self.weight * false_pos

        tversky_scores = \
            (true_pos + self.epsilon) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.epsilon)
        return 1 - tversky_scores.mean()


class KLLoss(torch.nn.Module):
    """KL Divergence loss"""
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def get_loss(
    loss_type: str,
    class_weights: Optional[torch.Tensor] = None,
    **params: dict
) -> torch.nn.Module:
    """Calculates the loss between the logits and target grids

    Args:
        loss_type: type of loss to use
        class_weigts: class weights

    Returns:
        torch.nn.Module: loss
    """
    if loss_type == "grid":
        return GridLoss(accuracy=params.get("accuracy", "absolute"))
    elif loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=params.get("gamma", 2))
    elif loss_type == "dice":
        return DiceLoss(weight=class_weights)
    elif loss_type == "tversky":
        return TverskyLoss(alpha=params.get("alpha", 0.3), beta=params.get("beta", 0.7), weight=class_weights)
    elif loss_type == "kl":
        return KLLoss()
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")


def combine_losses(
    losses: List[torch.Tensor], coeffs: List[float]
) -> torch.Tensor:
    return sum(
        [loss * coeff for loss, coeff in zip(losses, coeffs)]
    ) / len(losses)
