#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 21:11:26 2021

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class DenseBlock(nn.Module):
    def __init__(self , channel_in ,  channel_out,  init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d( channel_in  ,gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in  + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in  + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in  + 4 * gc, channel_out , 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5
        
class RoundStraightThrough(torch.autograd.Function):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, input):
        rounded = torch.round(input, out=None)
        return rounded

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

_round_straightthrough = RoundStraightThrough().apply
class StochasticRound(nn.Module):
    def __init__(self):
        super().__init__()
        self.hard_round = True

    def forward(self, x):
        u = torch.rand_like(x)

        h = x + u - 0.5

        if self.hard_round:
            h = _round_straightthrough(h)

        return h


class BackRound(nn.Module):

    def __init__(self):

        super().__init__()
        self.inverse_bin_width = 255.
        self.round_approx = 'stochastic'

        if self.round_approx == 'stochastic':
            self.round = StochasticRound()
        else:
            raise ValueError

    def forward(self, x):
        if self.round_approx == 'smooth' or self.round_approx == 'stochastic':
            h = x.clone()

            h = self.round(h)

            return h 

        else:
            raise ValueError  
 
class TorchRound(nn.Module):

    def __init__(self):

        super().__init__()
        self.round = torch.round

    def forward(self, x):
            h = x.clone()
            h = self.round(h)
            return h 

class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out = x * self.sigmoid(x)
        return out

    
class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size,
        bias=True, bn=False, res_scale=1):
        super(ResBlock, self).__init__()
        self.swish = Swish()
        act = nn.ReLU(True)
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res
    
class Encoder(nn.Module):
    def __init__(self,in_c,out_c,n_resblock, gc = 64 , bias = True):
        super(Encoder,self).__init__()
        # n_resblocks = 4
        self.conv1 = nn.Conv2d( in_c ,gc , 3 , 1 ,1 ,bias = bias)
        self.conv2 = nn.Conv2d( gc,gc , 3 , 1 ,1 ,bias = bias)
        self.conv3 = nn.Conv2d( gc , 2*gc , 3 ,1,1, bias = bias)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        resblock = [ ResBlock(gc,kernel_size=3) for _ in range(n_resblock)]
        self.resblock = nn.Sequential(*resblock)
    def forward(self,x):
        out = self.act(self.conv1(x))
        residual = out  
        out = self.resblock(out)
        out = self.conv2(out)
        out = out + residual
        out = self.conv3(out)
        return out

class Decoder(nn.Module):
    def __init__(self,out_c ,n_resblock, gc = 64 , bias = True):
        super(Decoder,self).__init__()        
        # n_resblocks = 4
        self.conv1 = nn.Conv2d( gc, gc , 3 , 1 ,1 ,bias = bias)
        resblock = [ ResBlock(gc,kernel_size=3) for _ in range(n_resblock)]
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.resblock = nn.Sequential(*resblock)
        self.conv2 = nn.Conv2d( gc , gc , 3 , 1 ,1 ,bias = bias)
        self.conv3 = nn.Conv2d( gc, out_c , 3 , 1 ,1 ,bias = bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self,x):
        out = self.act(self.conv1(x))
        residual = out
        out = self.resblock(out)
        out = self.conv2(out)
        out_ = out + residual
        out = self.conv3(out_)
        return residual , out_ , out
    
class HFVAE(nn.Module):
    '''
    Abstract : VAE using to genereate high frequency information
    Input    : Low frequency ( LR image)
    Ouput    : Generated High frequency information corresponding to input LR image
    
    '''
    def __init__(self, in_c , out_c):
        super(HFVAE, self).__init__()
        self.in_c = in_c # in_channel
        self.out_c = out_c # out_channel
        self.n_resblock = 4
        self.gc = 64
        self.encoder = Encoder(in_c,out_c,self.n_resblock , self.gc)
        self.decoder = Decoder(out_c,self.n_resblock , self.gc)

    def forward(self, x):
        latent = self.encoder(x)
        mu , logvar = torch.chunk(latent , 2 ,dim = 1)
        z = self.reparameterize(mu , logvar)
        gau_noise = torch.randn(z.shape).to(z.device)
        # print(z.shape)
        in_fea, out_fea, out = self.decoder(z)
        #print(out.shape)
        _ , out_gau_fea, out_gau = self.decoder(gau_noise)
        return z ,in_fea,  out_fea, out , out_gau  
    
    def encode(self,x):
        latent = self.encoder(x)
        # print(x.shape)
        mu , logvar = torch.chunk(latent , 2 ,dim = 1)
        return mu , logvar
    
    def decode(self , mu , logvar):
        z = self.reparameterize(mu , logvar)
        # print(z.shape)
        gau_noise = torch.randn(z.shape).to(z.device)
        in_fea, out_fea, out = self.decoder(z)
        _ , out_gau_fea , out_gau  = self.decoder(gau_noise)
        if self.opt['is_train']:
            self.rounding = BackRound()
        else:
            self.rounding = TorchRound()
        if self.opt['quantize_vae']:
            out = self.rounding(out*255.)/255.
            out_gau = self.rounding(out_gau*255.)/255.
        return z , in_fea, out_fea, out , out_gau        
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        # eps = torch.randn_like(mu)
        eps = torch.cuda.FloatTensor(std.size()).normal_()       
        eps = Variable(eps)
        
        return eps.mul(std).add_(mu)
    
    def kl_loss(self, mu, logvar, prior_mu=0):
        v_kl = -0.5*(logvar +1 -mu**2 - torch.exp(logvar))
        v_kl = torch.sum(v_kl)/ mu.shape[0]
        return v_kl        

class HFCVAE(nn.Module):
    '''
    Abstract : CVAE using to genereate high frequency information
    Input    : Low frequency ( LR image)
    Ouput    : Generated High frequency information corresponding to input LR image
    
    '''
    def __init__(self, in_c , out_c , opt):
        super(HFCVAE, self).__init__()
        self.in_c = in_c # in_channel
        self.out_c = out_c # out_channel
        self.opt = opt
        self.n_resblock = self.opt['n_resblock']
        self.device = torch.device('cuda' if self.opt['gpu_ids'] is not None else 'cpu')
        self.encoder = Encoder(3*2**self.opt['scale'],self.n_resblock ,self.opt )
        self.decoder = Decoder(out_c,self.n_resblock ,self.opt, gc = 64+3)

    def forward(self, LF,HF):
        out = torch.cat((LF,HF),dim=1)
        latent = self.encoder(out)
        mu , logvar = torch.chunk(latent , 2 ,dim = 1)
        z = self.reparameterize(mu , logvar)
        z = torch.cat((LF,z),dim=1)
        # print(z.shape)
        out = self.decoder(z)
        if self.opt['is_train']:
            self.rounding = BackRound()
        else:
            self.rounding = TorchRound()
        out = self.rounding(out*255.)/255.
        return mu , logvar , out       
    
    def encode(self,LF,HF):
        out = torch.cat((LF,HF),dim=1)
        latent = self.encoder(out)
        # print(x.shape)
        mu , logvar = torch.chunk(latent , 2 ,dim = 1)
        return mu , logvar
    
    def decode(self,LF):
        b,c,h,w = LF.shape
        z = torch.randn(size = [b, 64,h,w]).to(self.device)
        # z = self.reparameterize(mu , logvar)
        # print(z.shape)
        z = torch.cat((LF,z),dim=1)
        out = self.decoder(z)
        if self.opt['is_train']:
            self.rounding = BackRound()
        else:
            self.rounding = TorchRound()
        out = self.rounding(out*255.)/255.
        return out        
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        # eps = torch.randn_like(mu)
        eps = torch.cuda.FloatTensor(std.size()).normal_()       
        eps = Variable(eps)
        
        return eps.mul(std).add_(mu)
    
    def kl_loss(self, mu, logvar):
        v_kl = -0.5*(logvar +1 -mu**2 - torch.exp(logvar))
        v_kl = torch.sum(v_kl)/ mu.shape[0]
        return v_kl  
        
class DenseVAE(nn.Module):
    def __init__(self, in_c, out_c , opt):
        super(DenseVAE, self).__init__()
        self.in_c = in_c # in_channel
        self.out_c = out_c # out_channel
        self.opt = opt
        self.n_resblock = self.opt['n_resblock']
        self.gc = self.opt['gc']
        self.kl_annealing = 0
        self.device = torch.device('cuda' if self.opt['gpu_ids'] is not None else 'cpu')
        if not self.opt['decoder']:
            self.encoder = DenseBlock( self.in_c , self.out_c * 2)
        else:
            self.encoder = DenseBlock( self.in_c , self.gc * 2)
        if self.opt['decoder']:
            self.decoder = DenseBlock(self.gc , self.out_c)

    def forward(self, x):
        latent = self.encoder(x)
        mu , logvar = torch.chunk(latent , 2 ,dim = 1)
        z = self.reparameterize(mu , logvar)
        z_gau = torch.randn(mu.shape).to(x.device)
        # print(z.shape)
        if self.opt['is_train']:
            self.rounding = BackRound()
        else:
            self.rounding = TorchRound()
        if self.opt['decoder']:
            out = self.decoder(z)
            out_gau = self.decoder(z_gau)
        else:
            out = z
            out_gau = z_gau
        return self.rounding(out*255.)/255. , self.rounding(out_gau*255.)/255.
    
    def encode(self,x):
        latent = self.encoder(x)
        # print(x.shape)
        mu , logvar = torch.chunk(latent , 2 ,dim = 1)
        return mu , logvar
    
    def decode(self , mu , logvar):
        z = self.reparameterize(mu , logvar)
        z_gau = torch.randn(mu.shape).to(mu.device)
        if self.opt['is_train']:
            self.rounding = BackRound()
        else:
            self.rounding = TorchRound()
        if self.opt['decoder']:
            out = self.decoder(z)
            out_gau = self.decoder(z_gau)
        else:
            out = z
            out_gau = z_gau
        return self.rounding(z*255.)/255. , self.rounding(out*255.)/255. , self.rounding(out_gau*255.)/255.      
    
    def reparameterize(self, mu, logvar):
        if not self.opt['logistic']:
          std = torch.exp(0.5*logvar)
          eps = torch.cuda.FloatTensor(std.size()).normal_()       
          eps = Variable(eps)
          return eps.mul(std).add_(mu)
        else:
          y = torch.rand_like(mu)
          x = torch.exp(logvar) * torch.log(y / (1 - y) + 1e-7) + mu
          return x

    
    def kl_loss(self, mu, logvar, prior_mu=0):
        v_kl = -0.5*(logvar +1 -mu**2 - torch.exp(logvar))
        v_kl = torch.sum(v_kl)/ mu.shape[0]
        return v_kl            