import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class SimpleModel(nn.Module):
    def __init__(self, input_size = 64 * 64 * 3, output_size = 360):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

class ResNetModel(nn.Module):
    def __init__(self, num_classes=360):
        super(ResNetModel, self).__init__()
        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)