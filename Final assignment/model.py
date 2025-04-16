import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained weights
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        model = deeplabv3_resnet101(weights=weights)

        # Modify classifier head
        model.classifier[4] = nn.Conv2d(256, 19, kernel_size=1)
        nn.init.xavier_normal_(model.classifier[4].weight)

        # Set output stride to 16
        model.backbone.layer4[0].conv2.dilation = (2, 2)
        model.backbone.layer4[0].conv2.padding = (2, 2)
        model.backbone.layer4[0].downsample[0].stride = (1, 1)

        self.model = model  # store it

    def forward(self, x):
        return self.model(x)['out']