import torch.nn as nn
import torchvision.models as models

class UNet(nn.Module):
    def __init__(self, num_classes=2):
        super(UNet, self).__init__()
        self.base_model = models.segmentation.fcn_resnet18(pretrained=True)  # ✅ Use ResNet18 (smaller, faster)
        self.base_model.classifier[4] = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        return self.base_model(x)['out']

