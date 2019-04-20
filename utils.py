import sys
import time
import os

import torch
import torchvision
import torchvision.transforms.transforms as transforms
import torchvision.datasets as datasets

import numbers
import random
import math

import torchvision.transforms.functional as f


class Lighting(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def lighting(self, img, alphastd, eigval, eigvec):
        if not f._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
        img = f.to_tensor(img)
        alpha = img.new().resize_(3).normal_(0, alphastd)
        rgb = eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return f.to_pil_image(img.add(rgb.view(3, 1, 1).expand_as(img)))

    def __call__(self, img):
        if self.alphastd == 0:
            return img
        return self.lighting(img, self.alphastd, self.eigval, self.eigvec)


def get_dataloaders(datasetname, dataroot, batchsize, num_workers,
                    cropsize=None):
    if datasetname in ["cifar10", "cifar100"]:
        if datasetname == "cifar10":
            dataset = torchvision.datasets.CIFAR10
            mean = (125.3 / 255, 123.0 / 255, 113.9 / 255,)
            std = (63.0 / 255, 62.1 / 255, 66.7 / 255,)
        elif datasetname == "cifar100":
            dataset = torchvision.datasets.CIFAR100
            mean = (129.3 / 255, 124.1 / 255, 112.4 / 255)
            std = (68.2 / 255, 65.4 / 255, 70.4 / 255)

        transform_train = transforms.Compose([
            transforms.RandomCrop(cropsize or 32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        dataset_train = dataset(root=dataroot, train=True, download=True, transform=transform_train)
        dataset_test = dataset(root=dataroot, train=False, download=True, transform=transform_test)
        loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=batchsize, shuffle=True, num_workers=num_workers)
        loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=batchsize, shuffle=False, num_workers=num_workers)

    elif datasetname in ["ImageNet"]:
        traindir = os.path.join(dataroot, 'train')
        valdir = os.path.join(dataroot, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        pca_eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
        pca_eigvec = torch.Tensor([[-0.5675, 0.7192, 0.4009],
                                   [-0.5808, -0.0045, -0.8140],
                                   [-0.5836, -0.6948, 0.4203], ])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                Lighting(0.1, pca_eigval, pca_eigvec),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )

        loader_train = torch.utils.data.DataLoader(
            train_dataset, batch_size=batchsize, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        loader_test = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
            ),
            batch_size=batchsize, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    else:
        raise NotImplementedError("No such dataset:", datasetname)
    return loader_train, loader_test


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class ProgressBar:
    def __init__(self, total, msg=None):
        self.time_begin = time.time()
        self.count = 0
        self.total = total
        self.width = int(math.log10(self.total))
        if msg is not None:
            sys.stdout.write(msg + '\r')

    def update(self, msg=None):
        self.count += 1
        time_elapsed = time.time() - self.time_begin
        logs = []
        logs.append("  ")
        if msg:
            logs.append(msg)
            logs.append(", ")
        logs.append('{} ({}/itr), '.format(
            format_time(time_elapsed), format_time(time_elapsed / self.count)))
        logs.append('({:>{width}}/{:>{width}})'.format(self.count, self.total, width=self.width))
        if self.count < self.total:
            logs.append('\r')
        else:
            logs.append('\n')
        sys.stdout.write(''.join(logs))
        sys.stdout.flush()


def format_time(seconds):
    d = int(seconds // (60 * 60 * 24))
    h = int(seconds // (60 * 60) % 24)
    m = int(seconds // 60 % 60)
    s = int(seconds % 60)
    ms = int(seconds % 1.0 * 1000)
    if d > 0:
        return "{:02d}d{:02d}h".format(d, h)
    if h > 0:
        return "{:02d}h{:02d}m".format(h, m)
    if m > 0:
        return "{:02d}m{:02d}s".format(m, s)
    return "{:02d}s{:03d}".format(s, ms)
