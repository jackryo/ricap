from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import os
import re
import numpy as np
import argparse

import models
import utils
import trainers


def main():
    avaliable_modelnames = [m for m in dir(models)
                            if m[0] != '_' and type(getattr(models, m)).__name__ != 'module']
    parser = argparse.ArgumentParser(description='PyTorch RICAP Training')

    # hardware
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers loading data')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'ImageNet'],
                        help='dataset for training')
    parser.add_argument('--dataroot', type=str, default='data/',
                        help='path to dataset')

    # model
    parser.add_argument('--model', '-m', type=str, required=True, choices=avaliable_modelnames,
                        help='model name')
    parser.add_argument('--depth', '-d', type=int, required=True,
                        help='number of layers')
    parser.add_argument('--params', '-p', type=str, default=None,
                        help='model parameters such as widen factor for Wide ResNet')
    parser.add_argument('--postfix', type=str, default='',
                        help='postfix for saved model name')

    # hyperparameters
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='number of epochs: (default: 200 for Wide ResNet)')
    parser.add_argument('--batch', type=int, default=128,
                        help='batchsize')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='default learning rate')
    parser.add_argument('--droplr', type=float, default=0.2,
                        help='adaptive learning rate ratio: (default: 0.2 for Wide ResNet)')
    parser.add_argument('--adlr', type=str, default=None,
                        help='epochs at which learning rate is adapted (x droplr); e.g., \'60,120,160\' for Wide ResNet')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='weight decay: (default: 0.0005 for Wide ResNet)')

    # data augmentation
    parser.add_argument('--crop', type=int, default=None,
                        help='crop size')
    parser.add_argument('--beta_of_ricap', type=float, default=0.0,
                        help='beta of ricap augmentation')

    # save and resume
    parser.add_argument('--resume', '-r', type=int, default=0,
                        help='epoch at which resume from checkpoint. -1 for latest')
    parser.add_argument('--savefreq', type=int, default=5,
                        help='frequency to save model and to mark it the latest')
    parser.add_argument('--nocuda', action='store_true', default=False,
                        help='disable cuda devices.')
    args = parser.parse_args()

    print('==> Preparing dataset loaders..')
    dataloaders = utils.get_dataloaders(
        datasetname=args.dataset, dataroot=args.dataroot,
        batchsize=args.batch, num_workers=args.num_workers,
        cropsize=args.crop)

    # prepare log saving file name
    # save target : model information (.dat), result (.log), model parameters (.pth), optimizer parameters (.opt)
    savefilename_prefix = 'checkpoint/{model}-{depth}{params}_{dataset}{postfix}'.format(
        model=args.model,
        depth=args.depth,
        params='-{}'.format(args.params) if args.params is not None else '',
        dataset=args.dataset,
        postfix='_{}'.format(args.postfix) if args.postfix != '' else '',
    )

    # define learning rate strategy
    if args.adlr is None:
        args.adlr = np.array([60, 120, 160])
    else:
        assert re.match('[0-9 ,]+', args.adlr), 'Error: invalid adaptive learning rate: {}'.format(args.adlr)
        args.adlr = np.array(sorted(eval('[{}]'.format(args.adlr))))
    lr_current = args.lr

    # prepare cnn model and optimizer
    print('==> Building model..')
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    network = getattr(models, args.model)(args.dataset, args.depth, args.params)
    optimizer = optim.SGD(network.parameters(),
                          lr=lr_current, momentum=args.momentum, weight_decay=args.wd, nesterov=True)

    # write model information to save fine (.dat)
    with open('{}.dat'.format(savefilename_prefix), 'w') as of:
        print('==> Command', file=of)
        import sys
        print(' '.join(sys.argv), file=of)
        print('\n', file=of)
        print('==> Parameters', file=of)
        arg_str = '\n'.join(['--{} {}'.format(k, str(getattr(args, k))) for k in dir(args) if '_' not in k])
        print(arg_str, file=of)
        print('\n', file=of)
        print('==> Network', file=of)
        num_params = 0
        for param in network.parameters():
            num_params += param.numel()
        print('Number of parameters: %d' % num_params, file=of)
        print(network, file=of)

    # prepare trainer
    datasetname = args.dataset
    if datasetname == "cifar10":
        num_class = 10
    elif datasetname == "cifar100":
        num_class = 100
    elif datasetname == "ImageNet":
        num_class = 1000
    use_cuda = torch.cuda.is_available() and not args.nocuda
    trainer = trainers.make_trainer(
        network, dataloaders, optimizer, use_cuda=use_cuda, beta_of_ricap=args.beta_of_ricap)

    # initialize logs and epoch num
    if args.resume == 0:
        logs = []
        epoch_start = 0
    else:
        # if resuming
        # load model and optimizer parameter, start from pre-saved checkpoint
        print('==> Resuming from checkpoint..')
        if args.resume < 0:
            args.resume = 'latest'
        checkpoint = '{}_{}'.format(savefilename_prefix, args.resume)
        map_location = lambda storage, location: storage.cuda() if use_cuda else storage
        network.load_state_dict(torch.load(checkpoint + '.pth', map_location=map_location))
        optimizer.load_state_dict(torch.load(checkpoint + '.opt', map_location=map_location))
        logs = list(np.loadtxt(checkpoint + '.log', ndmin=2))
        epoch_start = len(logs)

    # update learning rate based on define learning rate strategy
    def update_learning_rate(epoch, ite):
        lr_adapted = args.lr * args.droplr**np.sum(args.adlr < epoch)
        if not lr_current == lr_adapted:
            print('Learning rate is adapted: {} -> {}'.format(lr_current, lr_adapted))
            utils.adjust_learning_rate(optimizer, lr_adapted)
        return lr_adapted

    # save network and optimizer parameter to save files (.pth, .opt)
    def savemodel(savefilename):
        torch.save(network.state_dict(), savefilename + '.pth')
        torch.save(optimizer.state_dict(), savefilename + '.opt')
        np.savetxt(savefilename + '.log', logs)

    # train and test loop
    epoch_end = args.epoch
    for epoch in range(epoch_start + 1, epoch_end + 1):
        lr_current = update_learning_rate(epoch, len(dataloaders[0]) * (epoch - 1))
        print('Epoch: {} / Iterations: {}'.format(epoch, len(dataloaders[0]) * (epoch - 1)))
        ret_train = trainer.epoch(train=True, lr=lr_current)
        ret_test = trainer.epoch(train=False, lr=lr_current)
        logs.append([epoch, ] + ret_train + ret_test + [lr_current, len(dataloaders[0]) * epoch])

        # save model and optimizer parameters
        if epoch % args.savefreq == 0 or epoch == epoch_end:
            print('Saving model as the latest..')
            savefilename = '{}_{}'.format(savefilename_prefix, 'latest')
            savemodel(savefilename)


if __name__ == '__main__':
    main()
