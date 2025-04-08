#This loss is adapted from https://github.com/shuaizzZ/Dice-Loss-PyTorch/blob/master/dice_loss.py 
# and inspired by https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68 
import torch
import torch.nn.functional as F
import torch.nn as nn

class DiceLoss(nn.Module):
    def __call__(self, prediction, target):
        #for Semantic segmentation predict has shape [N, C, H, W]
        pred = F.softmax(prediction, dim=1) #
        num_classes = pred.shape[1]  # Number of classes (C)
        dice = 0  # Initialize Dice loss accumulator

        for c in range(num_classes):
            pred_c = pred[:,c]
            target_c = target[:,c]

            intersection = (pred_c * target_c).sum(dim=(1, 2))  # Element-wise multiplication
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))  # Sum of all pixels
        
            dice += (2. * intersection + smooth) / (union + smooth)  # Per-class Dice score

        return 1-dice.mean()/num_classes

