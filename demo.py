import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict

import collections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    
    """
    # ***change mlp pretrained weight***
    # load mlp pretrained weight
    mlp_ckp = torch.load("/home/user/DISK2/data/yiting/representation/liif_modify/save/edsr_x2_representation/epoch-best.pth")['model']
    # build mlp
    mlp = model.imnet.state_dict()
    # Because the keys in pretrained weight and the keys in mlp are different, I need to create mlp_key here. ("imnet.")
    mlp_key = collections.OrderedDict()
    mlp_key['imnet.layers.0.weight'] = ''
    mlp_key['imnet.layers.0.bias'] = ''
    mlp_key['imnet.layers.2.weight'] = ''
    mlp_key['imnet.layers.2.bias'] = ''
    mlp_key['imnet.layers.4.weight'] = ''
    mlp_key['imnet.layers.4.bias'] = ''
    mlp_key['imnet.layers.6.weight'] = ''
    mlp_key['imnet.layers.6.bias'] = ''
    mlp_key['imnet.layers.8.weight'] = ''
    mlp_key['imnet.layers.8.bias'] = ''
    # I only want the last item of mlp_ckp which contains keys and values of mlp
    for n,i in mlp_ckp.items():
        #print(n)
        print(type(i))
        #print(i)
        print("done")
    # check the same key 
    state_dict = {k[6:]:v for k,v in i.items() if k in mlp_key.keys()} # [6:] -> delete "imnet."
    #print(state_dict)
    # update mlp
    mlp.update(state_dict)
    model.imnet.load_state_dict(mlp)
    print("weights are changed")
    """

    h, w = list(map(int, args.resolution.split(',')))
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(args.output)
