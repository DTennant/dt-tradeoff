import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import os
import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from dataset.cinic10 import get_cinic10_dataloader
from dataset.cifar10 import get_cifar10_dataloaders

from models import *


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--name', default='', type=str,
                        help='experiment name')
    parser.add_argument('--target_dataset', default='', type=str, choices=['cifar100', 'stl10', 'cinic10', 'svhn', 'tinyimagenet', 'emnist', 'fashion-mnist', 'cifar10'],
                        help='target dataset')  
    parser.add_argument('--source_dataset', default='cifar100', type=str, choices=['cifar100', 'stl10', 'cinic10', 'svhn', 'tinyimagenet', 'emnist', 'fashion-mnist', 'cifar10'],
                        help='source dataset')  
    parser.add_argument('--model', default='', type=str, choices=['wrn_28_4', 'ResNet18', 'ShuffleV2', 'resnext32_16x4d'],
                        help='model name')                    
    parser.add_argument('--ckpt_path', default='', type=str,
                        help='the path to ckpts trained on source dataset')
    parser.add_argument('--lr', default=4e-1, type=float,
                        help='learning rate')
    parser.add_argument('--epoch', default=30, type=int,
                        help='number of epochs')
    parser.add_argument('--batch', default=512, type=int,
                        help='batch size')
    parser.add_argument('--wd', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--start_epoch', default=10, type=int,
                        help='the start epoch model to be transferred')
    parser.add_argument('--skip', default=10, type=int,
                        help='conduct transfer experiment each N epoch')

    opt = parser.parse_args()

    batch_size = opt.batch

    if opt.target_dataset == 'cifar100':
        train_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        test_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        classes = 100

    elif opt.target_dataset == 'stl10':
        train_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        test_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        trainset = torchvision.datasets.STL10(root='./data', split='train',
                                                download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.STL10(root='./data', split='test',
                                            download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        classes = 10

    elif opt.target_dataset == 'cinic10':
        trainloader, testloader = get_cinic10_dataloader(batch_size=batch_size,
                                                                            num_workers=2,
                                                                            is_instance=False,
                                                                            augmentation='mine')
        classes = 10                                            

    elif opt.target_dataset == 'cifar10':
        trainloader, testloader = get_cifar10_dataloaders(batch_size=batch_size,
                                                                            num_workers=2,
                                                                            is_instance=False,
                                                                            augmentation='mine')
        classes = 10      

    elif opt.target_dataset == 'emnist':

        train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        trainset = torchvision.datasets.EMNIST(root='./data', split='byclass', train=True,
                                                download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.EMNIST(root='./data', split='byclass', train=False, 
                                            download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        classes = 10  

    elif opt.target_dataset == 'fashion-mnist':

        train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, 
                                            download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        classes = 10       

    else:
        raise NotImplementedError

    root = './save/model/{}/'.format(opt.ckpt_path)
    evaluated_epochs = list(range(501))[opt.start_epoch::opt.skip] # evaluate at most 500 epochs
    ckpts = [root + 'ckpt_epoch_{}.pth'.format(epoch) for epoch in evaluated_epochs]

    if opt.source_dataset == 'cifar100':
        source_num_classes = 100
    elif opt.source_dataset in ['stl10', 'cinic10', 'cifar10']:
        source_num_classes = 10
    else:
        raise NotImplementedError

    for j, ckpt in enumerate(ckpts):
        net = eval(opt.model)(num_classes=source_num_classes) 
        try:
            net.load_model(ckpt)
        except:
            break

        if 'wrn' in opt.model:
            net.fc = nn.Linear(net.fc.weight.shape[1], classes)
        elif 'resnext' in opt.model:
            net.classifier = nn.Linear(net.classifier.weight.shape[1], classes)
        elif 'ResNet' in opt.model or 'resnet' in opt.model:
            net.linear = nn.Linear(net.linear.weight.shape[1], classes)
        elif 'Shuffle' in opt.model:
            net.linear = nn.Linear(net.linear.weight.shape[1], classes)
        else:
            raise NotImplementedError

        net.cuda()
        net.train()
        criterion = nn.CrossEntropyLoss()

        trainable_params = []
        for k, v in net.named_parameters():
            if 'fc' in k or 'linear' in k or 'classifier' in k:
                trainable_params.append(v)
        optimizer = optim.SGD(trainable_params, lr=opt.lr, momentum=0.9, weight_decay=opt.wd)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch)

        best_acc = 0.
        for epoch in range(opt.epoch): 
            net.cuda()
            net.train()
            for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader), leave=False):
                net.cuda()

                if len(data) == 2:
                    inputs, labels = data
                else:
                    inputs, labels, _ = data
                inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            correct = 0
            total = 0
            net.cuda()
            net.eval()

            with torch.no_grad():
                for data in tqdm(testloader, leave=False):
                    if len(data) == 2:
                        images, labels = data
                    else:
                        images, labels, _ = data
                    images, labels = images.cuda(), labels.cuda()
                    outputs = net(images)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                acc = correct / total

            if acc > best_acc:
                best_acc = acc
                
            scheduler.step()
            print('Source Epoch: {}, Target Epoch: {}, Best Acc: {}'.format(j+1, epoch+1, best_acc))

    print('Finished Training')

if __name__ == "__main__":
    main()
