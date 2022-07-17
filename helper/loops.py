from __future__ import print_function, division

import sys
import time
import torch

from .util import AverageMeter, accuracy


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        if len(data) == 2:
            input, target = data
        elif len(data) == 3:
            input, target, index = data
        else:
            input, target, index, contrast_idx = data

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_ctc(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """CTC training"""
    # set modules as train()
    for module in module_list:
        module.train()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_ias = criterion_list[1]
    criterion_ips = criterion_list[2]

    model = module_list[0]
    info_bank = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    for idx, data in enumerate(train_loader):

        input, target, index, contrast_idx = data
        data_time.update(time.time() - end)

        input = input.float()
        input = input.cuda()

        target = target.cuda()
        index = index.cuda()
        contrast_idx = contrast_idx.cuda()
        
        # ===================forward=====================

        feat_s, logit_s = model(input, is_feat=True)
        
        if epoch <= opt.stage_two_epoch:
            loss_cls = criterion_cls(logit_s, target)
            output = criterion_ias(feat_s[-1], index)
            loss_ias = criterion_cls(output, index)
            loss_ias = opt.alpha * loss_ias.mean()
                
            loss_cls = loss_cls.mean()
            loss = loss_cls + loss_ias

        elif epoch > opt.stage_two_epoch:
            loss_cls = criterion_cls(logit_s, target)
            with torch.no_grad():
                feat_t, _ = info_bank(input, is_feat=True)
                feat_t = [f.detach() for f in feat_t]

            # other kd beyond KL divergence
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_ips = criterion_ips(f_s, f_t, index, contrast_idx)

            loss_cls = loss_cls.mean()
            loss_ips = opt.beta * loss_ips.mean()
            loss = loss_cls + loss_ips
        
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(val_loader):

            if len(data) == 2:
                input, target = data
            else:
                input, target, index = data
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target).mean()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        return top1.avg, top5.avg, losses.avg


def update_memory_bank(train_loader, model, criterion_list, opt):
    criterion_ips = criterion_list[2]
    # set modules as eval()
    model.eval()

    for idx, data in enumerate(train_loader):
        input, _, index, contrast_idx = data
        input = input.float()
        input = input.cuda()
        index = index.cuda()
        contrast_idx = contrast_idx.cuda()

        # ===================forward=====================

        with torch.no_grad():
            feat, _ = model(input, is_feat=True)
            gap_feat = feat[-1]
            criterion_ips(gap_feat, gap_feat, index, contrast_idx, is_update=True)
            if idx % opt.print_freq == 0:
                print('Updating momentum memory bank.')