import torch
import torch.nn as nn
from torchvision import models

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet50(weights=None)
        self.model.classifier[4] = nn.Conv2d(256, 19, kernel_size=1)
        nn.init.xavier_normal_(self.model.classifier[4].weight)

        # Modify backbone for output stride = 16
        self.model.backbone.layer4[0].conv2.dilation = (2, 2)
        self.model.backbone.layer4[0].conv2.padding = (2, 2)
        self.model.backbone.layer4[0].downsample[0].stride = (1, 1)

    def forward(self, x):
        return self.model(x)['out']