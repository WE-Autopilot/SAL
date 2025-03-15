import torch.nn as nn
import torch.nn.functional as F
import torch as pt
from torch.distributions import Normal


class SAL(nn.Module):
    def __init__(self, num_points=16, min_std=1e-4, max_std=2):
        super(SAL, self).__init__()
        self.num_points = num_points
        self.min_std = min_std
        self.max_std = max_std

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
            nn.Linear(1026, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4 * num_points + 1),
        )

    def forward(self, img, pos):
        x = self.conv_layers(img)  # Pass through convolutional layers
        x = x.mean((-2, -1))  # Global average pooling
        x = pt.cat((x, pos), dim=-1)
        x = self.fc_layers(x)  # Pass through fully connected layers

        mean = x[:, :2 * self.num_points]
        #mean[:, ::2] = pt.exp(mean[:, ::2] + 2)
        std = F.sigmoid(x[:, 2 * self.num_points:-1]) * self.max_std + self.min_std
        dist = Normal(mean, std)
        value = x[:, -1]

        return dist, value


if __name__ == "__main__":
    sal = SAL(16)
    img = pt.randn(16, 1, 512, 512)
    pos = pt.randn(16, 2)
    d, v = sal(img, pos)
    print(d.sample().shape, v.shape)
