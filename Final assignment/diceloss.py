#This loss is adapted from https://github.com/shuaizzZ/Dice-Loss-PyTorch/blob/master/dice_loss.py 
# and inspired by https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68 
import torch
import torch.nn.functional as F

Class DiceLoss(nn.Module):
    def __init__(self):
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def __call__(self, prediction, target):
        pred = F.softmax(prediction, dim=1)
        num_classes = 

