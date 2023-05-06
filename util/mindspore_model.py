import mindspore
from mindspore import nn
from mindspore.ops import operations as P


class LeNetMindspore(nn.Cell):
    def __init__(self):
        super(LeNetMindspore, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, pad_mode='valid')
        # Fully connected layers
        self.fc1 = nn.Dense(64, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)

    def construct(self, x):
        # 为x构造channel维度 1
        x = x.resize((x.shape[0], 1, x.shape[1], x.shape[2]))
        # Convolutional layers
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        # Fully connected layers
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x


class InceptionMindSpore(nn.Cell):
    def __init__(self, in_channels):
        super(InceptionMindSpore, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, pad_mode='pad', padding=2)
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, pad_mode='pad', padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, pad_mode='pad', padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def construct(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = P.ReduceMean(keep_dims=True)(x, (2, 3))
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return P.Concat(1)(outputs)


class LeNetProMindspore(nn.Cell):
    def __init__(self):
        super(LeNetProMindspore, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, pad_mode='valid')

        # Inception modules
        self.inception1 = InceptionMindSpore(in_channels=16)
        self.inception2 = InceptionMindSpore(in_channels=88)

        # Fully connected layers
        self.fc1 = nn.Dense(88, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)

    def construct(self, x):
        x = x.resize((x.shape[0], 1, x.shape[1], x.shape[2]))
        # Convolutional layers
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = self.inception1(x)
        x = self.inception2(x)
        # Fully connected layers
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x

