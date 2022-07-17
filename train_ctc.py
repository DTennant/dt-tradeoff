"""
the general training framework
"""

from __future__ import print_function

import os
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders_sample

from helper.loops import train_ctc as train, validate, update_memory_bank
from helper.util import adjust_learning_rate_decouple
from ctc.criterion import ContrastiveLoss
from ctc.instance import LinearAverage


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1000, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=300, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', help='where to decay lr, can be a list')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--weight_decay_coef', type=float, default=1.0, help='weight decay coefficient')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'svhn', 'tinyimagenet', 'cinic10'], help='dataset')
    parser.add_argument('--augmentation', default='vanilla', type=str, help='data augmentation type')

    # model
    parser.add_argument('--model', type=str, default='resnet8',
                        choices=['wrn_28_4', 'ResNet18', 'ResNet34', 'resnext32_16x4d', 'MobileNetV2', 'ShuffleV2'])
    parser.add_argument('--net_size', type=float, default=1.0, help='decay rate for learning rate')

    parser.add_argument('-a', '--alpha', type=float, default=0.1, help='weight for the [1] stage NCE loss')
    parser.add_argument('-b', '--beta', type=float, default=1.0, help='weight for the [2] stage NCE loss')

    # NCE 
    parser.add_argument('--instance_dim', default=128, type=int, help='feature dimension of stage [1]')
    parser.add_argument('--instance_t', default=0.1, type=float, help='temperature parameter for softmax of stage [1]')
    parser.add_argument('--instance_m', default=0.5, type=float, help='momentum for non-parametric updates of stage [1]')

    parser.add_argument('--stage_two_epoch', default=200, type=int, help='the epoch to end stage [1] and start stage [2]')
    parser.add_argument('--stage_two_decay_rate', default=0.1, type=float, help='adjust init learning rate of stage [2]')
    parser.add_argument('--update_memory_bank', action='store_true', help='update NCE memory before stage 2')

    parser.add_argument('--contrast_feat_dim', default=128, type=int, help='feature dimension of stage [2]')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE of stage [2]')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax of stage [2]')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates of stage [2]')

    # tags
    parser.add_argument('--note', default='', type=str, help='note for special experiment')

    opt = parser.parse_args()

    opt.model_path = './save/model'

    opt.model_name = '{}_{}_a:{}_b:{}_{}'.format(opt.model, opt.dataset, opt.alpha, opt.beta, opt.note)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():

    opt = parse_option()
    opt.stage_two_epochs = opt.epochs - opt.stage_two_epoch # for calculating learning rate scheduler


    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model initialization
    info_bank = model_dict[opt.model](num_classes=n_cls)
    model = model_dict[opt.model](num_classes=n_cls)

    best_acc = 0.

    data = torch.randn(2, 3, 32, 32)
    
    model.eval()
    with torch.no_grad():
        feat, _ = model(data, is_feat=True)

    model = model.cuda()
    module_list = nn.ModuleList([])
    module_list.append(model)

    criterion_cls = nn.CrossEntropyLoss(reduce=False).cuda()
    opt.feat_dim = feat[-1].shape[1]
    opt.n_data = n_data

    criterion_ips = ContrastiveLoss(opt).cuda()
    module_list.append(criterion_ips.embed.cuda())
    module_list.append(criterion_ips.embed_info_bank.cuda())
    info_bank = info_bank.cuda()
    
    criterion_ias = LinearAverage(opt.feat_dim, opt.instance_dim, n_data, opt.instance_t, opt.instance_m).cuda()

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    
    criterion_list.append(criterion_ias)    
    criterion_list.append(criterion_ips)     

    # optimizer
    optimizer = optim.SGD(module_list.parameters(),
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay * opt.weight_decay_coef)

    module_list.append(info_bank)

    cudnn.benchmark = True

    # routine
    for epoch in range(1, opt.epochs + 1):

        lr = adjust_learning_rate_decouple(epoch, opt, optimizer, warm_up_epochs1=0, warm_up_epochs2=0)

        # print("==> training...")
        if epoch == opt.stage_two_epoch + 1 and opt.update_memory_bank:
            update_memory_bank(train_loader, info_bank, criterion_list, opt)
            print('memory bank refreshed.')

        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)

        test_acc, _, _ = validate(val_loader, model, criterion_cls, opt)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

        if epoch == opt.stage_two_epoch:
            info_bank.load_state_dict(model.state_dict())
            del module_list[-1]

            module_list.append(info_bank)
            print('start training stage [2]')

            for group in optimizer.param_groups:
                group['weight_decay'] = opt.weight_decay


if __name__ == '__main__':
    main()
