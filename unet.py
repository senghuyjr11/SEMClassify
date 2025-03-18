import torch.nn as nn

from torchvision import models
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.base_model = models.segmentation.fcn_resnet50(pretrained=True)  # ✅ Use ResNet50
        self.base_model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        return self.base_model(x)['out']

