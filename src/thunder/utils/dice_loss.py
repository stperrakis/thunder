import torch.nn.functional as F


# Adapted from https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
def multiclass_dice_loss(pred, target, smooth=1):
    """
    Computes Dice Loss for multi-class segmentation.
    :param pred: Tensor of predictions (batch_size, C, H, W).
    :param target: One-hot encoded ground truth (batch_size, C, H, W).
    :param smooth: Smoothing factor.

    :return: Scalar Dice Loss.
    """
    pred = F.softmax(pred, dim=1)  # Convert logits to probabilities
    num_classes = pred.shape[1]  # Number of classes (C)
    dice = 0  # Initialize Dice loss accumulator

    for c in range(num_classes):  # Loop through each class
        pred_c = pred[:, c]  # Predictions for class c
        target_c = (target == c).long()  # Ground truth for class c

        intersection = (pred_c * target_c).sum()  # Element-wise multiplication
        union = pred_c.sum() + target_c.sum()  # Sum of all pixels

        dice += (2.0 * intersection + smooth) / (union + smooth)  # Per-class Dice score

    return 1 - dice.mean() / num_classes  # Average Dice Loss across classes
