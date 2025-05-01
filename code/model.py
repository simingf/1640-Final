import torch
import torch.nn as nn
import torch.nn.functional as F

class RotationModel(nn.Module):
    def __init__(self, num_classes=36):
        super(RotationModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=32, 
                               kernel_size=3, 
                               stride=1, 
                               padding="same")
        self.conv2 = nn.Conv2d(in_channels=32, 
                               out_channels=64, 
                               kernel_size=3, 
                               stride=1, 
                               padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, 
                             out_features=128)
        self.fc2 = nn.Linear(in_features=128, 
                             out_features=num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x