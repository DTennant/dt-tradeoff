import torch
from torch import nn
from .memory import ContrastMemory

eps = 1e-7


class ContrastiveLoss(nn.Module):
    """Contrastive Loss function"""

    def __init__(self, opt):
        super(ContrastiveLoss, self).__init__()
        self.embed = Embed(opt.feat_dim, opt.contrast_feat_dim)
        self.embed_info_bank = Embed(opt.feat_dim, opt.contrast_feat_dim)
        self.contrast = ContrastMemory(opt.contrast_feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion = ContrastLoss(opt.n_data)
        self.criterion_info_bank = ContrastLoss(opt.n_data)

    def refresh(self, opt):
        self.__init__(opt)

    def forward(self, f, f_info_bank, idx, contrast_idx=None, is_sharpen=False, is_update=False):
        if is_update:
            f_info_bank = self.embed_info_bank(f)
            f = self.embed(f)
            self.contrast.update_memory(f, f_info_bank, idx)
        else:
            f = self.embed(f)
            f_info_bank = self.embed_info_bank(f_info_bank, is_sharpen)
            out, out_info_bank = self.contrast(f, f_info_bank, idx, contrast_idx)
            loss = self.criterion(out) + self.criterion_info_bank(out_info_bank)
            return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x, is_sharpen=False):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
