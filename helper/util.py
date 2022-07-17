from __future__ import print_function

import torch
import numpy as np
import torch


def adjust_learning_rate(epoch, opt, optimizer, warm_up_epochs=0):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if epoch <= warm_up_epochs:
        rate = 1.0 * epoch / warm_up_epochs
        for param_group in optimizer.param_groups:
            new_lr = rate * opt.learning_rate
            param_group['lr'] = new_lr
    if (epoch-1) in np.asarray(opt.lr_decay_epochs):
        for param_group in optimizer.param_groups:
            new_lr = param_group['lr'] * opt.lr_decay_rate
            param_group['lr'] = new_lr
    return optimizer.param_groups[0]['lr']


def adjust_learning_rate_decouple(epoch, opt, optimizer, warm_up_epochs1=5, warm_up_epochs2=5):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    if epoch <= opt.stage_two_epoch:
        if epoch <= warm_up_epochs1:
            rate = 1.0 * epoch / warm_up_epochs1
            for param_group in optimizer.param_groups:
                new_lr = rate * opt.learning_rate
                param_group['lr'] = new_lr
        else:
            rate = 0.5 * (1 + np.cos(epoch * np.pi / (opt.stage_two_epoch + 1))) # +1 or not???????
            for param_group in optimizer.param_groups:
                new_lr = opt.learning_rate * rate
                param_group['lr'] = new_lr
    else:
        epoch_ = epoch - opt.stage_two_epoch
        if epoch_ <= warm_up_epochs2:
            rate = 1.0 * epoch_ / warm_up_epochs2
            for param_group in optimizer.param_groups:
                new_lr = opt.stage_two_decay_rate * opt.learning_rate * rate
                param_group['lr'] = new_lr
        else:
            rate = 0.5 * (1 + np.cos(epoch_ * np.pi / (opt.stage_two_epochs + 1)))
            for param_group in optimizer.param_groups:
                new_lr = opt.stage_two_decay_rate * opt.learning_rate * rate
                param_group['lr'] = new_lr

    return optimizer.param_groups[0]['lr']


def adjust_learning_rate_cosine(epoch, opt, optimizer, warm_up_epochs=5, min_lr_scale=0.001):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    if epoch <= warm_up_epochs:
        rate = 1.0 * epoch / warm_up_epochs
        for param_group in optimizer.param_groups:
            new_lr = rate * opt.learning_rate
            param_group['lr'] = new_lr
    else:
        rate = min_lr_scale + (1.0 - min_lr_scale) * 0.5 * (1 + np.cos(epoch / (opt.epochs + 1) * np.pi))
        for param_group in optimizer.param_groups:
            new_lr = opt.learning_rate * rate
            param_group['lr'] = new_lr

    return optimizer.param_groups[0]['lr']


def mixup_data(x, y, alpha=0.5, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float()
            correct_k = correct_k.sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':

    pass
