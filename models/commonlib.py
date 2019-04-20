import torch.nn as nn


def get_num_classes(dataset):
    return {
        "cifar10": 10,
        "cifar100": 100,
        "ImageNet": 1000,
    }[dataset]


class Vectorize(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
