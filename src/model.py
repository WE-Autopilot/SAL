import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_points=16):
        super(Model, self).__init__()
        self.num_points = num_points

        # Convolutional layers using Sequential
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Fully connected layers using Sequential
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * num_points),
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Pass through convolutional layers
        x = x.mean((-2, -1))  # Global average pooling
        x = self.fc_layers(x)  # Pass through fully connected layers
        return x
