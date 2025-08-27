import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

import numpy as np
import matplotlib.pyplot as plt
from models.CTM_model import HFVAE

from models.positional_embedding import Embedding

#@register('liif_frequency_with_bilinear_coord_modulation_new_3d')
@register('asfsr_wo_freq')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)
        """
        self.feat_dim = 8
        self.netCTM = HFVAE(3, self.feat_dim)
        """
        self.pe_in_channels = 2
        self.pe_n_freqs = 10
        self.pe = Embedding(in_channels=self.pe_in_channels, N_freqs=self.pe_n_freqs)
        self.pe_dim = self.pe_in_channels*(2*self.pe_n_freqs+1)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            """
            # self.feat_dim
            imnet_in_dim += self.feat_dim
            """
            if self.feat_unfold:
                imnet_in_dim *= 9
            #imnet_in_dim += self.pe_dim # attach coord PE(global)
            imnet_in_dim += 4 # attach coord (local/relative/global)
            if self.cell_decode:
                imnet_in_dim += 2
            
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim, 'coord_dim': self.pe_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        # LR bilinear upsampling
        self.inp = inp
        
        self.feat = self.encoder(inp)
        """
        _ , _ , _ , self.freq, _ = self.netCTM(inp)
        
        self.feat = torch.cat([self.feat, self.freq], dim=1)
        #print(self.feat.shape)
        """
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                
                bs, q = coord.shape[:2]
                pe_global_coord = self.pe(coord.view(bs * q, -1)).view(bs, q, -1)
                #inp = torch.cat([q_feat, pe_global_coord], dim=-1)
                inp = torch.cat([q_feat, rel_coord, q_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    
                    inp = torch.cat([inp, rel_cell], dim=-1)

                #bs, q = coord.shape[:2]
                #pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                pred = self.imnet(inp.contiguous().view(bs * q, -1), pe_global_coord.contiguous().view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
            
        # LR bilinear upsampling
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
              padding_mode='border', align_corners=False)[:, :, 0, :] \
              .permute(0, 2, 1)
        
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
