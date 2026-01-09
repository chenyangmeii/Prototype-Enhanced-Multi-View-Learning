import torch
import torch.nn as nn
from torchvision.models import resnet as tv_resnet
from .mixstyle import MixStyle


class Model(nn.Module):

    def __init__(self, num_classes=2, pretrained=True, p=0.5, alpha=0.1):
        super(Model, self).__init__()

        backbone = tv_resnet.resnet18(pretrained=pretrained)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.classifier = nn.Linear(512, num_classes)


        self.mixstyle1 = MixStyle(p=p, alpha=alpha, mix="random")
        self.mixstyle2 = MixStyle(p=p, alpha=alpha, mix="random")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.mixstyle1(x)

        x = self.layer2(x)
        x = self.mixstyle2(x)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
