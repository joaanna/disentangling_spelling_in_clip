import torch
import torch.nn.functional as F
import sys

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class LinearSubspace(torch.nn.Module):
    def __init__(self, out_dim, input_size=512, device='cuda:0'):
        super(LinearSubspace, self).__init__()
        self.input_size = input_size
        self.out_dim = out_dim
        self.W = torch.nn.Parameter(torch.randn(self.out_dim, self.input_size), requires_grad=True)
        self.device = device
        torch.nn.init.orthogonal_(self.W)


    def forward(self, img_i, img_t, img_ti, txt_i, txt_t, class_id, weight=None):
        if weight is None:
            weight = self.W
        img_i = F.linear(img_i, weight)
        img_i = img_i / img_i.norm(dim=-1, keepdim=True)
        img_t = F.linear(img_t, weight)
        img_t = img_t / img_t.norm(dim=-1, keepdim=True)
        img_ti = F.linear(img_ti, weight)
        img_ti = img_ti / img_ti.norm(dim=-1, keepdim=True)
        txt_i = F.linear(txt_i, weight)
        txt_i = txt_i / txt_i.norm(dim=-1, keepdim=True)
        txt_t = F.linear(txt_t, weight)
        txt_t = txt_t / txt_t.norm(dim=-1, keepdim=True)
        return (img_i, img_t, img_ti, txt_i, txt_t)


    def _weight_norm(self, device):
        return torch.norm(torch.eye(self.out_dim).to(device) - self.W @ self.W.t())


def symmetric_cross_entropy(img, txt, device='cpu', weight=1.):
    logits_per_image = weight * img @ txt.t()
    logits_per_text = weight * txt @ img.t()
    n = logits_per_image.shape[0]
    loss_i = F.cross_entropy(logits_per_image, torch.arange(n).to(device))
    loss_t = F.cross_entropy(logits_per_text, torch.arange(n).to(device))
    return 0.5*(loss_i+loss_t)