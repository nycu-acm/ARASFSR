"""
SISN
Created by: Yuanzhi Wang
Email: w906522992@gmail.com
Core Link: https://github.com/mdswyz/SISN-Face-Hallucination
Paper Link: https://dl.acm.org/doi/10.1145/3474085.3475682
"""
from argparse import Namespace
from models import register
from .utils import common
from .utils.isab import Internal_feature_Split_Attention_Block
import torch.nn as nn
import math
import torch

class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range, sign=-1,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0),
    ):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, num_channels, res_scale=1.0):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, num_channels, scale):
        m = list()
        if (scale & (scale-1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m += [nn.Conv2d(num_channels, 4*num_channels, 3, 1, 1)]
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m += [nn.Conv2d(num_channels, 9*num_channels, 3, 1, 1)]
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError

        super().__init__(*m)


class DownBlock(nn.Module):
    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h//self.scale, self.scale, w//self.scale, self.scale)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * (self.scale**2), h//self.scale, w//self.scale)
        return x

## External-internal Split Attention Group (ESAG)
class External_internal_Split_Attention_Group(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_ISABs, conv2 = common.default_conv):
        super(External_internal_Split_Attention_Group, self).__init__()
        modules_body = []
        modules_body = [
            conv(
                n_feat, n_feat) \
            for _ in range(n_ISABs)]
        modules_body.append(conv2(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)
        self.IA = Internal_feature_Split_Attention_Block(n_feat,n_feat)

    def forward(self, x):
        res = self.body(x)
        x = self.IA(x)
       
        res += x
        return res

## Split Attention in Split_Attention Network (SISN)
class Net(nn.Module):
    def __init__(self, args, conv = Internal_feature_Split_Attention_Block, conv2 = common.default_conv):
        super(Net, self).__init__()
        self.args = args
        n_ESAGs = args.num_groups
        n_ISABs = args.num_blocks
        n_feats = args.num_channels
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(255, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [
            DownBlock(args.scale),
            nn.Conv2d(3 * args.scale ** 2, args.num_channels, 3, 1, 1),
        ]

        # define body module
        modules_body = [
            External_internal_Split_Attention_Group(
                conv, n_feats, kernel_size, n_ISABs=n_ISABs) \
            for _ in range(n_ESAGs)]

        modules_body.append(conv2(n_feats, n_feats, kernel_size))
        
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        
        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = 3
            # define tail module
            modules_tail = [
                common.Upsampler(conv2, scale, n_feats, act=False),
                conv2(n_feats, 3, kernel_size)]
            self.tail = nn.Sequential(*modules_tail)

        self.add_mean = common.MeanShift(255, rgb_mean, rgb_std, 1)

    def forward(self, x):
        #x = self.sub_mean(x)

        x = self.head(x)

        res = self.body(x)
        res += x
        
        if self.args.no_upsampling:
            x = res
        else:
            x = self.tail(res)
        
        #x = self.add_mean(x)

        return x 

@register('sisn')
def make_sisn(num_blocks=10, res_scale=1.0, no_upsampling=False):
    args = Namespace()
    
    args.num_groups = 10
    args.num_blocks = num_blocks
    args.num_channels = 64
    args.reduction = 16
    args.res_scale = res_scale
    args.scale = 1
    args.no_upsampling = no_upsampling
    
    return Net(args)