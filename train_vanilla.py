from __future__ import print_function

import os
import argparse
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders
from dataset.cifar10 import get_cifar10_dataloaders
from dataset.cinic10 import get_cinic10_dataloader

from helper.util import adjust_learning_rate_cosine
from helper.loops import train_vanilla as train, validate


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=500, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--min_lr_scale', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,150,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='ResNet18',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_16_4', 'wrn_40_1', 'wrn_40_2', 'wrn_64_4',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnext32_16x4d',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                                 'wrn_28_2', 'wrn_28_4', 'wrn_40_4', 'wrn_28_10', 'wrn_16_10', 'wrn_22_4', 'wrn_34_4', 'wrn_46_4', 'wrn_52_4'])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'tinyimagenet', 'cifar10', 'cinic10'], help='dataset')
    parser.add_argument('--augmentation', type=str, default='none', help='data augmentation type')

    parser.add_argument('-t', '--trial', type=str, default='0', help='the experiment id')

    parser.add_argument('--notes', default='', type=str, help='notes for special experiments')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.model_path = './save/model'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_trial_{}_{}'.format(opt.model, opt.dataset, opt.learning_rate, opt.trial, opt.notes)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    best_acc = 0

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader, _ = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=True, augmentation=opt.augmentation)
        n_cls = 100

    else:
        raise NotImplementedError(opt.dataset)


    # model
    model = model_dict[opt.model](num_classes=n_cls)
    model = nn.DataParallel(model)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # routine
    for epoch in range(1, opt.epochs + 1):

        lr = adjust_learning_rate_cosine(epoch, opt, optimizer, min_lr_scale=opt.min_lr_scale)

        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        test_acc, _, test_loss = validate(val_loader, model, criterion, opt)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state_dict = model.state_dict()
            state = {
                'epoch': epoch,
                'model': state_dict,
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state_dict = model.state_dict()
            state = {
                'epoch': epoch,
                'model': state_dict,
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
