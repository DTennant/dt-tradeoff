# -*- coding: utf-8 -*-
from __future__ import division

""" 
Creates a ResNeXt Model as defined in:
Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor, is_last=False):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)
        self.is_last = is_last

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        out = residual + bottleneck
        if self.is_last:
            return F.relu(out, inplace=True), out
        else:
            return F.relu(out, inplace=True)


class CifarResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, cardinality, depth, base_width, widen_factor=4, num_classes=100):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(CifarResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 6
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = num_classes
        self.output_size = 64
        self.stages = [32, 32 * self.widen_factor, 64 * self.widen_factor, 128 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(32)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.classifier = nn.Linear(self.stages[3], self.nlabels)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.base_width, self.widen_factor))
            else:
                block.add_module(name_,
                                 ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width,
                                                   self.widen_factor, is_last=(bottleneck == self.block_depth - 1)))
        return block

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def load_model(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = pretrain_dict["model"] if "model" in pretrain_dict else pretrain_dict
        from collections import OrderedDict

        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if k in model_dict:
                # print('gotcha {}'.format(k))
                new_dict[k] = v
            # if 'block1' in k:
                # print('gotcha {}'.format(k.replace('block3', 'block_adapt')))
                # new_dict[k.replace('block1', 'block_adapt')] = v
            # if 'conv1' in k and 'block' not in k:
                # print('gotcha {}'.format(k.replace('block3', 'block_adapt')))
                # new_dict[k.replace('conv1', 'conv_adapt')] = v
            # if 'fc' in k:
                # print('gotcha {}'.format(k.replace('block3', 'block_adapt')))
                # new_dict[k.replace('fc', 'fc_adapt')] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")

    def get_bn_before_relu(self):
        bn1 = self.layer1[-1].bn2
        bn2 = self.layer2[-1].bn2
        bn3 = self.layer3[-1].bn2

        return [bn1, bn2, bn3]

    def forward(self, x, is_feat=False, preact=False):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        f0 = x
        x, f1_pre = self.stage_1.forward(x)
        f1 = x
        x, f2_pre = self.stage_2.forward(x)
        f2 = x
        x, f3_pre = self.stage_3.forward(x)
        f3 = x
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(-1, self.stages[3])
        f4 = x

        x = self.classifier(x)

        if is_feat:
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4], x
            else:
                return [f0, f1, f2, f3, f4], x
        else:
            return x
        # if is_feat:
        #     if
        # return self.classifier(x)

def resnext32_4x16d(**kwargs):
    return CifarResNeXt(4, 32, 16, 4, **kwargs)

def resnext32_8x8d(**kwargs):
    return CifarResNeXt(8, 32, 8, 4, **kwargs)

def resnext32_16x4d(**kwargs):
    return CifarResNeXt(16, 32, 4, 4, **kwargs)

def resnext32_32x2d(**kwargs):
    return CifarResNeXt(32, 32, 2, 4, **kwargs)