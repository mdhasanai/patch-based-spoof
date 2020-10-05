import torch
from torch.nn import functional as F
import math

class PadSameConv(torch.nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride, pad_size):
        super(PadSameConv, self).__init__()
        self.pad_size = pad_size
        self.conv = torch.nn.Conv2d(in_size, out_size, kernel_size, stride)

    def forward(self, x):
        out = F.pad(x, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), 'circular')
        out = self.conv(out)
#         print(out.shape)
        return out


class ChannelBasedLayer(torch.nn.Module):
    def __init__(self, input_size, depth_size):
        super(ChannelBasedLayer, self).__init__()
        output_size = 150
        step = 50
        modules = []
        cnt = depth_size // 50 - 2
        for _ in range(cnt):
            modules.append(PadSameConv(input_size, output_size, kernel_size=3, stride=1, pad_size=1))
            modules.append(torch.nn.BatchNorm2d(output_size))
            modules.append(torch.nn.MaxPool2d(2, 2))
            modules.append(torch.nn.ReLU())
            input_size += step
            output_size += step

        self.sequential = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.sequential(x)


class PatchModel(torch.nn.Module):
    def __init__(self, im_size, channel_size=3):
        self.depth_size = self.get_depth_size(im_size)
        super(PatchModel, self).__init__()

#         self.conv1 = PadSameConv(3, 50, kernel_size=5, stride=1, pad_size=2)
        self.conv1 = torch.nn.Conv2d(channel_size, 50, kernel_size=5, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(50)
        self.maxpool1 = torch.nn.MaxPool2d(2, 2)

        self.conv2 = torch.nn.Conv2d(50, 100, kernel_size=3, stride=1)
        self.bn2 = torch.nn.BatchNorm2d(100)
        self.maxpool2 = torch.nn.MaxPool2d(2, 2)

        self.conv3 = ChannelBasedLayer(100, self.depth_size)

        self.fc1 = torch.nn.Linear(self.depth_size * 3 * 3, 1000)
        self.bn6 = torch.nn.BatchNorm1d(1000)
        self.fc2 = torch.nn.Linear(1000, 400)
        self.bn7 = torch.nn.BatchNorm1d(400)
        self.fc3 = torch.nn.Linear(400, 2)

        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.relu = torch.nn.ReLU()

    def get_depth_size(self, im_size):
        depth_size = int(math.log(im_size // 3, 2)) * 50
        return depth_size

    def forward(self, x):
        out = F.pad(x, (2, 2, 2, 2), 'circular')
        out = self.bn1(self.conv1(out))
        out = self.maxpool1(out)
        out = self.relu(out)

        out = F.pad(out, (1, 1, 1, 1), 'circular')
        out = self.maxpool2(self.bn2(self.conv2(out)))
        out = self.relu(out)
        out = self.conv3(out)
        out = out.view(-1, self.depth_size * 3 * 3)

        out = self.bn6(self.fc1(out))
        out = self.relu(out)

        out = self.bn7(self.fc2(out))
        
        out = self.relu(out)

        out = self.fc3(out)
        out = self.logsoft(out)

        return out
    
    
