#This loss is adapted from https://github.com/shuaizzZ/Dice-Loss-PyTorch/blob/master/dice_loss.py 
import torch
import torch.nn.functional as F
import torch.nn as nn

class MultiDiceLoss(nn.Module):
    def __init__(self, ignore_index=255, epsilon=1e-5):
        super(MultiDiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, prediction, target):
        N,C,H,W = prediction.shape # prediction: [N, C, H, W]
        prediction = F.softmax(prediction, dim=1) 
        dice = 0

        for i in range(C):
            target_mask = (target==i).float()
            output_c = prediction[:,i,:,:]
            intersection = (output_c*target_mask).sum()
            union = target_mask.sum()+output_c.sum()
            dice += (2*intersection+self.epsilon)/(union+self.epsilon)

        return 1-dice/C









        


