#This loss is adapted from https://github.com/shuaizzZ/Dice-Loss-PyTorch/blob/master/dice_loss.py 
import torch
import torch.nn.functional as F
import torch.nn as nn

class MultiDiceLoss(nn.Module):
    def __init__(self, ignore_index=255, epsilon=1e-6):
        super(MultiDiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, prediction, target):
        N, C, H, W = prediction.shape  # prediction: [N, C, H, W]
        prediction = F.softmax(prediction, dim=1)

        # Create mask to ignore 'ignore_index' pixels
        valid_mask = (target != self.ignore_index).unsqueeze(1).float()  # [N,1,H,W]

        # One-hot encode the target
        target_one_hot = F.one_hot(target.clamp(0, C-1), num_classes=C)  # [N,H,W,C]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()      # [N,C,H,W]

        # Apply valid mask to both prediction and target
        prediction = prediction * valid_mask  # mask out invalid pixels
        target_one_hot = target_one_hot * valid_mask

        # Compute Dice loss
        intersection = (prediction * target_one_hot).sum(dim=(0,2,3))  # per class
        union = prediction.sum(dim=(0,2,3)) + target_one_hot.sum(dim=(0,2,3))  # per class
        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)

        return 1 - dice.mean()









        


