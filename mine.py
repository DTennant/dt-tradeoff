import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from models import *
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.stl10 import get_stl10_dataloaders
from dataset.cinic10 import get_cinic10_dataloader


class MINE(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, 1)
            nn.init.normal_(self.fc1.weight,std=0.02)
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.normal_(self.fc2.weight,std=0.02)
            nn.init.constant_(self.fc2.bias, 0)
            nn.init.normal_(self.fc3.weight,std=0.02)
            nn.init.constant_(self.fc3.bias, 0)
            
        def forward(self, inputs):
            output = F.relu(self.fc1(inputs))
            output = F.relu(self.fc2(output))
            output = F.relu(self.fc3(output))
            output = self.fc4(output)
            return output


def sample_batch(data, first_dim, batch_size, sample_mode='joint'):
    if sample_mode == 'joint':
        index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[index]
    else:
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        second_dim = data.shape[1] - first_dim
        batch = np.concatenate([data[joint_index][:,0:first_dim].reshape(-1,first_dim), data[marginal_index][:,first_dim:].reshape(-1,second_dim)],
                                    axis=1)
    return batch


def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et


def learn_mine(batch, mine_net, mine_net_optim,  ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint , marginal = batch
    joint = torch.FloatTensor(joint).cuda()
    marginal = torch.FloatTensor(marginal).cuda()
    mi_lb , t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
    
    # unbiasing use moving average
    loss = -(torch.mean(t) - torch.log(torch.mean(et)) * (et.mean().detach()) / ma_et.detach())

    # use biased estimator
    if torch.isnan(loss):
        return 0, 0, 0
    
    mine_net_optim.zero_grad()
    loss.backward()
    mine_net_optim.step()
    return mi_lb, ma_et, loss


def train_xt(data, mine_net, mine_net_optim, first_dim=3072, batch_size=5000, iter_num=int(10e+3), log_freq=int(2e+3), seq=0):
    # data is x or y
    iter_num = int(iter_num)
    result = list()
    ma_et = 1.
    for i in range(iter_num):
        batch = sample_batch(data,batch_size=batch_size, first_dim=first_dim)\
        , sample_batch(data,batch_size=batch_size,sample_mode='marginal', first_dim=first_dim)
        mi_lb, ma_et, loss = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        if mi_lb == 0 and ma_et == 0:
            continue
        result.append(mi_lb.detach().cpu().numpy())
        if (i + 1) % log_freq == 0:
            print(result[-1])
    return result


def get_xty_batches(model, loader, num_classes):
    model.eval();
    model.cuda();
    x_batches = []
    t_batches = []
    y_batches = []

    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
    
        output = model(inputs, is_feat=True)
    
        x_batches.append(inputs.reshape((inputs.shape[0], -1)).cpu().numpy())
        t_batches.append(output[0][-1].reshape(inputs.shape[0], -1).detach().cpu().numpy())
        y_batches.append(np.eye(num_classes)[labels.cpu().numpy()])
        
    x_batches = np.concatenate(x_batches)
    t_batches = np.concatenate(t_batches)
    y_batches = np.concatenate(y_batches)
    
    xt_batches = np.concatenate((x_batches, t_batches), axis=1)
    ty_batches = np.concatenate((t_batches, y_batches), axis=1)
    return xt_batches, ty_batches


def ma(a, window_size=500):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]


def get_mi(opt, a_dim, b_dim, ab_batches, first_dim=3072, return_res=False, seq=0):
    mine_net = MINE(input_size=a_dim + b_dim, hidden_size=opt.hidden_size).cuda()
    mine_optim = torch.optim.Adam(mine_net.parameters(), lr=opt.lr)
    results = train_xt(ab_batches, mine_net, mine_optim, first_dim=first_dim, iter_num=opt.iter_num, seq=seq)

    if return_res:
        return ma(results)[-1]
    # estimated MI
    mis = []
    for it in range(100):
        batch = sample_batch(ab_batches, first_dim=a_dim), sample_batch(ab_batches, first_dim=a_dim, sample_mode='marginal')
        joint, marginal = batch
        joint, marginal = torch.from_numpy(joint).float().cuda(), torch.from_numpy(marginal).float().cuda()
        mi, _, _ = mutual_information(joint, marginal, mine_net)
        mis.append(mi.detach().item())
    return sum(mis) / len(mis)


def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', default='xt', type=str, choices=['xt', 'ty'],
                        help='calculate I(X;T) or I(T;Y)')                    
    parser.add_argument('--model', default='', type=str,
                        help='network architecture name')  
    parser.add_argument('--ckpt_path', default='', type=str,
                        help='path to checkpoints trained on source data')                    
    parser.add_argument('--iter_num', default=5e+3, type=float,
                        help='param for training MINE')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='param for training MINE')
    parser.add_argument('--hidden_size', default=1024, type=int,
                        help='param for training MINE')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='number of data loader workers')
    parser.add_argument('--start_epoch', default=10, type=int,
                        help='the start epoch model to be transferred')
    parser.add_argument('--skip', default=10, type=int,
                        help='conduct transfer experiment each N epoch')
    parser.add_argument('--dataset', default='cifar100', type=str,
                        help='target dataset')
    parser.add_argument('--source_dataset', default='cifar100', type=str,
                        help='source dataset')
    opt = parser.parse_args()

    if opt.dataset == 'cifar100':
        _, testloader = get_cifar100_dataloaders(batch_size=512, num_workers=opt.num_workers, augmentation='mine')
        num_classes = 100
    elif opt.dataset == 'stl10':
        _, testloader = get_stl10_dataloaders(batch_size=512, num_workers=opt.num_workers,)
        num_classes = 10
    elif opt.dataset == 'cinic10':
        _, testloader = get_cinic10_dataloader(batch_size=512, num_workers=opt.num_workers,)
        num_classes = 10
    else:
        raise NotImplementedError(opt.dataset)

    root = './save/model/{}/'.format(opt.ckpt_path)
    evaluated_epochs = list(range(501))[opt.start_epoch::opt.skip]
    ckpts = [root+'ckpt_epoch_{}.pth'.format(epoch) for epoch in evaluated_epochs]

    if opt.source_dataset == 'cifar100':
        source_num_classes = 100
    elif opt.source_dataset in ['stl10', 'cinic10', 'cifar10']:
        source_num_classes = 10
    else:
        raise NotImplementedError
    model = eval(opt.model)(num_classes=source_num_classes)

    data = torch.randn(2, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        feat, _ = model(data, is_feat=True)

    x_dim = 3 * 32 * 32
    t_dim = feat[-1].shape[1]
    y_dim = num_classes

    count = 0
    for i, ckpt in enumerate(ckpts):
        epoch = evaluated_epochs[i]

        if opt.mode == 'xt':
            r = model.load_model(ckpt)
            xt_batches, _ = get_xty_batches(model, testloader, num_classes)
            avg_mi = get_mi(opt, x_dim, t_dim, xt_batches, first_dim=x_dim, return_res=True, seq=count)
            count += 1
            print(f'I(X;T) of {opt.model} of {epoch}-th epoch is : {avg_mi}')

        if opt.mode == 'ty':
            r = model.load_model(ckpt)
            _, ty_batches = get_xty_batches(model, testloader, num_classes)
            avg_mi = get_mi(opt, t_dim, y_dim, ty_batches, first_dim=t_dim, return_res=True, seq=count)
            count += 1
            print(f'I(T;Y) of {opt.model} of {epoch}-th epoch is : {avg_mi}')
            
if __name__ == "__main__":
    main()