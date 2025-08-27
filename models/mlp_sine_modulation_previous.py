import torch.nn as nn
import torch.nn.functional as F
from models import register

import torch

@register('mlp_sine_modulation_previous')

class MLP(nn.Module):

        
    def __init__(self, in_dim, coord_dim, out_dim, hidden_list):
        super().__init__()
        self.I = nn.ModuleList()
        self.C = nn.ModuleList()
        for hidden in hidden_list:
            self.I.append(nn.Sequential(nn.Linear(in_dim, hidden)))
            self.C.append(nn.Sequential(nn.Linear(coord_dim, hidden)))
            in_dim = hidden * 2
            coord_dim = hidden
        self.output_linear = nn.Linear(hidden, out_dim)

    def forward(self, x, coord):
        shape = x.shape[:-1]
        feat = self.I[0](x)
        feat = F.relu(feat)
        coord = self.C[0](coord)
        coord = torch.sin(coord)
        mod = feat * coord
        #print(feat_in.shape)
        #print(mod.shape)
        for i in range(1, len(self.I)):
            feat = self.I[i](torch.cat([feat, mod], dim=1))
            feat = F.relu(feat)
            coord = self.C[i](coord)
            coord = torch.sin(coord)
            mod = feat * coord
        x = self.output_linear(mod)
        return x.view(*shape, -1)
        