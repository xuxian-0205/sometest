import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlockttc(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlockttc, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))


        return self.relu(x + y)

class TTCEncoder(nn.Module):
    def __init__(self,input_dim = 484, output_dim=256, norm_fn='batch', dropout=0.0):
        super(TTCEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.in_planes = input_dim  #559
        self.layer1 = self._make_layer(int(output_dim/2), stride=1)  #64
        self.layer2 = self._make_layer(int(output_dim/4), stride=1)  #32
        self.layer3 = self._make_layer(int(output_dim/8), stride=1)   #16

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(int(output_dim/8), int(output_dim/8), kernel_size=1)  # 16

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer0 = nn.Conv2d(self.in_planes, dim, kernel_size=1)
        layer1 = ResidualBlockttc(dim, dim, self.norm_fn, stride=stride)
        layers = (layer0, layer1)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        xall = []
        x = self.layer1(x)
        xall.append(x)
        x = self.layer2(x)
        xall.append(x)
        x = self.layer3(x)
        xall.append(x)
        x = self.conv2(x)
        xall.append(x)

        x = torch.cat([xn for xn in xall],dim=1)
        return x
class maskHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(maskHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.Sigmod = nn.Sigmoid()
    def forward(self, x):
        return self.Sigmod(self.conv2(self.relu(self.conv1(x))))
class midHead(nn.Module):
    def  __init__(self, input_dim=128, hidden_dim=256):
        super(midHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim,kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()  
    def forward(self, x):
        return  self.tanh(self.conv2(self.relu(self.conv1(x))))

class flowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(flowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim,kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class preHead(nn.Module):
    def __init__(self, input_dim=128,out_dim=128, hidden_dim=256):
        super(preHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_dim ,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))