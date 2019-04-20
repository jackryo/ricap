import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import numpy as np
import utils


class Trainer(nn.Module):
    def __init__(self, network, dataloaders, optimizer, use_cuda=False):
        super(Trainer, self).__init__()
        self.network = network
        self.loader_train, self.loader_test = dataloaders
        self.optimizer = optimizer

        if use_cuda:
            self.nGPUs = torch.cuda.device_count()
            print('==> Transporting model to {} cuda device(s)..'.format(self.nGPUs))
            if self.nGPUs > 1:
                self.network = nn.DataParallel(self.network, device_ids=range(self.nGPUs))
            self.network.cuda()
            self.cuda = lambda x: x.cuda()
            cudnn.benchmark = True
        else:
            self.cuda = lambda x: x
            print('==> Keeping all on CPU..')

    def epoch(self, train=False, lr=0.1):
        if train:
            self.network.train()
            loader = self.loader_train
            forward = self.forward_train
        else:
            self.network.eval()
            loader = self.loader_test
            forward = self.forward_test

        loss_total = 0
        sample_error = 0
        sample_error5 = 0
        sample_total = 0
        progress = utils.ProgressBar(len(loader), '<progress bar is initialized.>')

        for batch_idx, (inputs, targets) in enumerate(loader):
            batchsize = targets.size(0)
            outputs, loss_batch = forward(inputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted5 = torch.topk(outputs.data, 5)
            sample_total += batchsize
            sample_error += batchsize - predicted.cpu().eq(targets).sum().item()
            loss_total += loss_batch.data.item() * batchsize
            loss = float(loss_total / sample_total)
            err = float(1. * sample_error / sample_total)
            result = predicted5[:, 0].cpu().eq(targets)
            for i in range(4):
                result += predicted5[:, i + 1].cpu().eq(targets)
            result = result.sum().item()
            sample_error5 += batchsize - result
            err5 = float(1. * sample_error5 / sample_total)

            progress.update(
                '{}, top1 loss: {:0.4f}, err:{:5.2f}% ({:5d}/{:5d}), top5 err:{:5.2f}% ({:5d}/{:5d}), lr:{}'.format(
                    'train' if train else ' test', loss, 100 * err,
                    int(sample_error), int(sample_total), 100 * err5,
                    int(sample_error5), int(sample_total), lr))

        return [err, loss]

    def forward_train(self, inputs, targets):
        self.optimizer.zero_grad()
        inputs = Variable(self.cuda(inputs))
        targets = Variable(self.cuda(targets))
        outputs = self.network(inputs)
        loss_batch = F.cross_entropy(outputs, targets)
        loss_batch.backward()
        self.optimizer.step()
        return outputs, loss_batch

    def forward_test(self, inputs, targets):
        with torch.no_grad():
            inputs = Variable(self.cuda(inputs))
        with torch.no_grad():
            targets = Variable(self.cuda(targets))
        with torch.no_grad():
            outputs = self.network(inputs)
        loss_batch = F.cross_entropy(outputs, targets)
        return outputs, loss_batch


class TrainerRICAP(Trainer):

    def __init__(self, network, dataloaders, optimizer, beta_of_ricap, use_cuda=False):
        super(TrainerRICAP, self).__init__(
            network, dataloaders, optimizer, use_cuda)
        self.beta = beta_of_ricap

    def ricap(self, images, targets):

        beta = self.beta  # hyperparameter

        # size of image
        I_x, I_y = images.size()[2:]

        # generate boundary position (w, h)
        w = int(np.round(I_x * np.random.beta(beta, beta)))
        h = int(np.round(I_y * np.random.beta(beta, beta)))
        w_ = [w, I_x - w, w, I_x - w]
        h_ = [h, h, I_y - h, I_y - h]

        # select four images
        cropped_images = {}
        c_ = {}
        W_ = {}
        for k in range(4):
            index = self.cuda(torch.randperm(images.size(0)))
            x_k = np.random.randint(0, I_x - w_[k] + 1)
            y_k = np.random.randint(0, I_y - h_[k] + 1)
            cropped_images[k] = images[index][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
            c_[k] = targets[index]
            W_[k] = (w_[k] * h_[k]) / (I_x * I_y)

        # patch cropped images
        patched_images = torch.cat(
            (torch.cat((cropped_images[0], cropped_images[1]), 2),
             torch.cat((cropped_images[2], cropped_images[3]), 2)),
            3)

        targets = (c_, W_)
        return patched_images, targets

    def ricap_criterion(self, outputs, c_, W_):
        loss = sum([W_[k] * F.cross_entropy(outputs, Variable(c_[k])) for k in range(4)])
        return loss

    def forward_train(self, inputs, targets):
        self.optimizer.zero_grad()
        inputs, targets = self.cuda(inputs), self.cuda(targets)
        inputs, (c_, W_) = self.ricap(inputs, targets)
        inputs = Variable(inputs)
        outputs = self.network(inputs)
        loss_batch = self.ricap_criterion(outputs, c_, W_)
        loss_batch.backward()
        self.optimizer.step()
        return outputs, loss_batch


def make_trainer(network, dataloaders, optimizer, use_cuda, beta_of_ricap=0.0):
    if beta_of_ricap:
        return TrainerRICAP(network, dataloaders, optimizer, beta_of_ricap, use_cuda)
    else:
        return Trainer(network, dataloaders, optimizer, use_cuda)
