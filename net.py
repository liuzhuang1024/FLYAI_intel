# ## build CNN
# from torch import nn
#
#
# ## build CNN
# class Net(nn.Module):
#     # def __init__(self,num_classes=10):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
#         self.relu1 = nn.ReLU(True)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
#         self.relu2 = nn.ReLU(True)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
#         self.relu3 = nn.ReLU(True)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
#         self.relu4 = nn.ReLU(True)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.pool4 = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(128 * 8 * 8, 1024)
#         self.relu5 = nn.ReLU(True)
#         self.fc2 = nn.Linear(1024, 6)
#
#     def forward(self, input):
#         output = self.conv1(input)
#         output = self.relu1(output)
#         output = self.bn1(output)
#         output = self.pool1(output)
#
#         output = self.conv2(output)
#         output = self.relu2(output)
#         output = self.bn2(output)
#         output = self.pool2(output)
#
#         output = self.conv3(output)
#         output = self.relu3(output)
#         output = self.bn3(output)
#         output = self.pool3(output)
#
#         output = self.conv4(output)
#         output = self.relu4(output)
#         output = self.bn4(output)
#         output = self.pool4(output)
#
#         output = output.view(-1, 128 * 8 * 8)
#         output = self.fc1(output)
#         output = self.relu5(output)
#         output = self.fc2(output)
#
#         return output
import torch
from torch import nn


# 因为ResNet34包含重复的单元，故用ResidualBlock类来简化代码
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),  # 要采样的话在这里改变stride
            nn.BatchNorm2d(outchannel),  # 批处理正则化
            nn.ReLU(inplace=True),  # 激活
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),  # 采样之后注意保持feature map的大小不变
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)  # 计算残差
        out += residual
        return nn.ReLU(inplace=True)(out)  # 注意激活


# ResNet类
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )  # 开始的部分
        self.body = self.makelayers([3, 4, 35, 3])  # 具有重复模块的部分
        self.classifier = nn.Linear(512, 6)  # 末尾的部分

    def makelayers(self, blocklist):  # 注意传入列表而不是解列表
        self.layers = []
        for index, blocknum in enumerate(blocklist):
            if index != 0:
                shortcut = nn.Sequential(
                    nn.Conv2d(64 * 2 ** (index - 1), 64 * 2 ** index, 1, 2, bias=False),
                    nn.BatchNorm2d(64 * 2 ** index)
                )  # 使得输入输出通道数调整为一致
                self.layers.append(ResidualBlock(64 * 2 ** (index - 1), 64 * 2 ** index, 2, shortcut))  # 每次变化通道数时进行下采样
            for i in range(0 if index == 0 else 1, blocknum):
                self.layers.append(ResidualBlock(64 * 2 ** index, 64 * 2 ** index, 1))
        return nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.body(x)
        x = nn.AvgPool2d(3)(x)  # kernel_size为7是因为经过多次下采样之后feature map的大小为7*7，即224->112->56->28->14->7
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


net = Net()
print(net)
