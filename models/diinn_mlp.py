import torch.nn as nn
import torch.nn.functional as F
from models import register

import torch


@register('diinn_mlp')

class MLP(nn.Module):

        
    def __init__(self, in_dim, coord_dim, out_dim, hidden_list):
        super().__init__()
        self.K = nn.ModuleList()
        self.Q = nn.ModuleList()
        feat_dim = in_dim
        for hidden in hidden_list:
            self.K.append(nn.Sequential(nn.Linear(feat_dim, hidden)))
            self.Q.append(nn.Sequential(nn.Linear(coord_dim, hidden)))
            feat_dim = in_dim + hidden
            coord_dim = hidden
        self.output_linear = nn.Linear(hidden, out_dim)

    def forward(self, x, coord):
        shape = x.shape[:-1]
        feat_in = self.K[0](x)
        feat_in = F.relu(feat_in)
        coord = self.Q[0](coord)
        coord = torch.sin(coord)
        mod = feat_in * coord
        #print(feat_in.shape)
        #print(mod.shape)
        for i in range(1, len(self.K)):
            feat = self.K[i](torch.cat([mod, x], dim=1))
            feat = F.relu(feat)
            coord = self.Q[i](mod)
            coord = torch.sin(coord)
            mod = feat * coord
        x = self.output_linear(mod)
        return x.view(*shape, -1)
        