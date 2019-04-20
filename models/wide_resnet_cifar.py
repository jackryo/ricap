import torch
import torch.nn as nn
import torch.nn.init as init
from .commonlib import *
import math

__all__ = [
    'WideResNet',
    'WideResNetDropout',
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


class BasicWideBlock(nn.Module):
    def __init__(self, c_in, c_out, stride, dropout=0.0):
        super(BasicWideBlock, self).__init__()

        # pre-activation layer
        self.pre_act = nn.Sequential(
            nn.BatchNorm2d(c_in, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )

        # main conv layers ( conv-bn-relu )
        self.block = nn.Sequential(*[
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        ] + ([nn.Dropout(p=dropout)] if dropout > 0.0 else []) + [
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        ])

        # residual shortcut
        if c_in != c_out:
            self.shortcut = nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        o1 = self.pre_act(x)
        o = self.block(o1)
        shortcut = self.shortcut(o1) if hasattr(self, 'shortcut') else x
        return o + shortcut

# conv block


class Group(nn.Module):
    def __init__(self, c_in, c_out, n, stride, dropout=0.0):
        super(Group, self).__init__()
        sequential = []
        for i in range(n):
            if i == 0:
                sequential.append(BasicWideBlock(c_in, c_out, stride, dropout=dropout))
            else:
                sequential.append(BasicWideBlock(c_out, c_out, 1, dropout=dropout))
        self.f = nn.Sequential(*sequential)

    def forward(self, x):
        return self.f(x)


def WideResNet(dataset, depth, params, dropout=0.0):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    if params is None:
        print("widen factor isset to 10 as default value")
        width = 10
    else:
        width = int(params)

    widths = torch.Tensor([16, 32, 64]).mul(width).int()
    num_classes = get_num_classes(dataset)

    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
        Group(16, widths[0].item(), n, stride=1, dropout=dropout),
        Group(widths[0].item(), widths[1].item(), n, stride=2, dropout=dropout),
        Group(widths[1].item(), widths[2].item(), n, stride=2, dropout=dropout),
        nn.BatchNorm2d(widths[2].item(), eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        Vectorize(),
        nn.Linear(widths[2].item(), num_classes),
    )
    init_params(model)
    return model


def WideResNetDropout(dataset, depth, params):
    return WideResNet(dataset, depth, params, dropout=0.3)
