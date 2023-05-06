import torch
from torch import nn
from torch.nn import functional as F


class LeNetTorch(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetTorch, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pool(F.relu(self.conv1(x)))  # 6 * 12 * 12
        x = self.pool(F.relu(self.conv2(x)))  # 16 * 4 * 4
        x = x.view(batch_size, -1, 16 * 4 * 4)  #
        x = F.relu(self.fc1(x))  # 120
        x = F.relu(self.fc2(x))  # 84
        x = self.fc3(x)  # 10
        return x


class InceptionTorch(nn.Module):
    def __init__(self, in_channels):
        super(InceptionTorch, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class LeNetProTorch(nn.Module):
    """在LeNet的最后一层卷积后加入两个inception模块"""

    def __init__(self, num_classes=10):
        super(LeNetProTorch, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)

        # Inception modules
        self.inception1 = InceptionTorch(16)
        self.inception2 = InceptionTorch(88)

        # Fully connected layer
        self.fc1 = nn.Linear(88 * 16, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))  # 6 * 24 * 24
        x = F.max_pool2d(x, 2, 2)  # 6 * 12 * 12
        x = F.relu(self.conv2(x))  # 16 * 8 * 8
        x = F.max_pool2d(x, 2, 2)  # 16 * 4 * 4

        # Inception modules
        x = self.inception1(x)  # 48 * 4 * 4
        x = self.inception2(x)  # 48 * 4 * 4

        # Flatten
        x = x.view(batch_size, -1)
        # Fully connected layer
        x = F.relu(self.fc1(x))  # 120
        x = self.fc2(x)  # 10
        return x


