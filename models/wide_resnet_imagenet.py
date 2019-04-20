import torch
import torch.nn as nn
import torch.nn.init as init
from .commonlib import *
import math

__all__ = [
    'WideResNetBottleneck',
    'WideResNetDropoutBottleneck',
]


def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            init.uniform_(m.weight, -stdv, stdv)
            if m.bias is not None:
                init.constant_(m.bias, 0)


class Shortcut(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super(Shortcut, self).__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(int(c_out), eps=1e-5, momentum=0.1),
        )

    def forward(self, x):
        return self.shortcut(x)


class WideBlockBottleneck(nn.Module):
    def __init__(self, c_in, c_out, stride, dropout=0.0):
        super(WideBlockBottleneck, self).__init__()

        # main conv layers ( conv-bn-relu )
        self.block = nn.Sequential(*[
            nn.Conv2d(c_in, int(c_out / 2), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(c_out / 2), eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(c_out / 2), int(c_out / 2), kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(int(c_out / 2), eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(c_out / 2), c_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(c_out), eps=1e-5, momentum=0.1),
        ] + ([nn.Dropout(p=dropout)] if dropout > 0.0 else [])
        )
        self.ReLU = nn.ReLU(inplace=True)

        # residual shortcut
        if stride != 1 or c_in != c_out:
            self.shortcut = Shortcut(c_in, c_out, stride)

    def forward(self, x):
        o = self.block(x)
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = o + shortcut
        out = self.ReLU(out)
        return out

# conv block


class Group(nn.Module):
    def __init__(self, c_in, c_out, n, stride, dropout=0.0):
        super(Group, self).__init__()
        sequential = []
        for i in range(n):
            if i == 0:
                sequential.append(WideBlockBottleneck(c_in, c_out, stride, dropout=dropout))
            else:
                sequential.append(WideBlockBottleneck(c_out, c_out, 1, dropout=dropout))
        self.f = nn.Sequential(*sequential)

    def forward(self, x):
        return self.f(x)


def WideResNetBottleneck(dataset, depth, params, dropout=0.0):
    if params is None:
        print("widen factor isset to 2 as default value")
        width = 2
    else:
        width = int(params)

    widths = torch.Tensor([128, 256, 512, 1024]).mul(width).int()
    num_classes = get_num_classes(dataset)
    if depth == 50:
        blocks = torch.Tensor([3, 4, 6, 3]).int()
    else:
        raise NotImplementedError("define num of layers")
        # blocks = torch.Tensor([, , , ]).int()

    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        Group(64, widths[0], blocks[0], stride=1, dropout=dropout),
        Group(widths[0], widths[1], blocks[1], stride=2, dropout=dropout),
        Group(widths[1], widths[2], blocks[2], stride=2, dropout=dropout),
        Group(widths[2], widths[3], blocks[3], stride=2, dropout=dropout),
        nn.AdaptiveAvgPool2d(1),
        Vectorize(),
        nn.Linear(widths[3], num_classes),
    )
    init_params(model)
    return model


def WideResNetDropoutBottleneck(dataset, depth, params):
    return WideResNetBottleneck(dataset, depth, params, dropout=0.3)
