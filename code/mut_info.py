import torch.nn as nn
import torch.nn.functional as F


class DownStudent(nn.Module):
    def __init__(self, from_=32, to_=3):
        super(DownStudent, self).__init__()
        self.from_ = from_
        self.to_ = to_

        self.conv1 = nn.Conv2d(32, 16, kernel_size=1, padding=2)
        self.conv1_bn = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 8, kernel_size=2, padding=5)
        self.conv2_bn = nn.BatchNorm2d(8)

        self.conv3 = nn.Conv2d(8, 3, kernel_size=2, padding=9)
        self.conv3_bn = nn.BatchNorm2d(3)

    def forward(self, x):
        out = x
        if self.from_ >= 32:
            out = F.relu(self.conv1_bn(self.conv1(out)))

        if self.to_ == 16:
            return out

        if self.from_ >= 16:
            out = F.relu(self.conv2_bn(self.conv2(out)))

        if self.to_ == 8:
            return out

        out = self.conv2_bn(self.conv3(out))

        return out


class UpStudent(nn.Module):
    def __init__(self, from_=3, to_=64):
        super(UpStudent, self).__init__()
        self.from_ = from_
        self.to_ = to_

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(in_features=128, out_features=64)

    def forward(self, x):
        out = x
        if self.from_ <= 3:
            out = F.relu(self.conv1_bn(self.conv1(out)))
            out = F.max_pool2d(out, 2)

        if self.to_ == 8:
            return out

        if self.from_ <= 8:
            out = F.relu(self.conv2_bn(self.conv2(out)))
            out = F.max_pool2d(out, 2)

        if self.to_ == 16:
            return out

        if self.from_ <= 16:
            out = self.conv3_bn(self.conv3(out))
            out = F.max_pool2d(out, 2)

        if self.to_ == 32:
            return out

        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out


class Student_Teacher(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Student_Teacher, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1)
        self.conv1_bn = nn.BatchNorm2d(2 * in_channels)

        self.conv2 = nn.Conv2d(2 * in_channels, 2 * out_channels, kernel_size=1)
        self.conv2_bn = nn.BatchNorm2d(2 * out_channels)

        self.conv3 = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1)

    def forward(self, x):

        out = F.relu(self.conv1_bn(self.conv1(x)))

        out = F.relu(self.conv2_bn(self.conv2(out)))

        out = self.conv3(out)

        return out


class Linear_Model(nn.Module):
    def __init__(self, in_size, out_size):
        super(Linear_Model, self).__init__()
        self.lin = nn.Linear(in_size, out_size)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.lin(out)
        return out


class Mutual_Info(nn.Module):
    def __init__(self, sequence):
        super(Mutual_Info, self).__init__()
        self.sequence = nn.ModuleList(sequence)

    def forward(self, x):
        for module in self.sequence:
            x = module(x)
        return x
