import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Original Author: Wei Yang
"""

__all__ = ['wrn']


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, is_last=False, is_student=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.is_last = is_last

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        pre1 = self.bn2(self.conv1(out if self.equalInOut else x))
        out = self.relu2(pre1)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, is_student=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, is_student)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, is_student):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, i == nb_layers - 1, is_student))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, groups_split=1, ratio=1.0, is_student=False):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # self.conv_adapt = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
        #                        padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, is_student=False)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, is_student=False)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, is_student=False)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.is_student = is_student

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
            else:
                assert 'what initialization do you want??????'

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.block1)
        feat_m.append(self.block2)
        feat_m.append(self.block3)
        return feat_m

    def get_bn_before_relu(self, is_student=False):
        if is_student:
            bn3 = self.bn1

            return [bn3, bn3, bn3]
        else:
            bn1 = self.block2.layer[0].bn1
            bn2 = self.block3.layer[0].bn1
            bn3 = self.bn1

            return [bn1, bn2, bn3]

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
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")

    def load_fc(self, pretrain):
        print("Loading pretrain FC from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = pretrain_dict["model"] if "model" in pretrain_dict else pretrain_dict
        from collections import OrderedDict

        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if 'fc' in k:
                print('gotcha {}'.format('fc'))
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Student classifier has been loaded......")

    def load_backbone(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = pretrain_dict["model"] if "model" in pretrain_dict else pretrain_dict
        from collections import OrderedDict

        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if 'fc' in k:
                pass
            elif k in model_dict:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)

    def forward(self, x, is_feat=False, preact=False):
        out = self.conv1(x)
        f0 = out
        out = self.block1(out)
        f1 = out
        out = self.block2(out)
        f2 = out
        out = self.block3(out)
        f3 = out
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        f4 = out
        out = self.fc(out)
        if is_feat:
            if preact:
                f1 = self.block2.layer[0].bn1(f1)
                f2 = self.block3.layer[0].bn1(f2)
                f3 = self.bn1(f3)
            return [f0, f1, f2, f3, f4], out
        else:
            return out


def wrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model


def wrn_40_2(**kwargs):
    model = WideResNet(depth=40, widen_factor=2, **kwargs)
    return model


def wrn_40_1(**kwargs):
    model = WideResNet(depth=40, widen_factor=1, **kwargs)
    return model


def wrn_16_2(**kwargs):
    model = WideResNet(depth=16, widen_factor=2, **kwargs)
    return model


def wrn_16_1(**kwargs):
    model = WideResNet(depth=16, widen_factor=1, **kwargs)
    return model

def wrn_28_4(**kwargs):
    model = WideResNet(depth=28, widen_factor=4, **kwargs)
    return model

def wrn_28_2(**kwargs):
    model = WideResNet(depth=28, widen_factor=2, **kwargs)
    return model

def wrn_16_4(**kwargs):
    model = WideResNet(depth=16, widen_factor=4, **kwargs)
    return model

def wrn_22_4(**kwargs):
    model = WideResNet(depth=22, widen_factor=4, **kwargs)
    return model

def wrn_34_4(**kwargs):
    model = WideResNet(depth=34, widen_factor=4, **kwargs)
    return model

def wrn_40_4(**kwargs):
    model = WideResNet(depth=40, widen_factor=4, **kwargs)
    return model

def wrn_46_4(**kwargs):
    model = WideResNet(depth=46, widen_factor=4, **kwargs)
    return model

def wrn_52_4(**kwargs):
    model = WideResNet(depth=52, widen_factor=4, **kwargs)
    return model

def wrn_64_4(**kwargs):
    model = WideResNet(depth=64, widen_factor=4, **kwargs)
    return model

def wrn_28_10(**kwargs):
    model = WideResNet(depth=28, widen_factor=10, **kwargs)
    return model

def wrn_16_10(**kwargs):
    model = WideResNet(depth=16, widen_factor=10, **kwargs)
    return model


if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 32, 32)
    net = wrn_40_2(num_classes=100)
    feats, logit = net(x, is_feat=True, preact=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')
