import argparse
import os
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict

"""
python demo_scface.py --model "/mnt/HDD4/yiting/liif_modify/load/SCface_SR_MM/diinn_edsr_real.pth" --gpu 1

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--resolution', default='128,128')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    
    
    # file(file)
    h, w = list(map(int, args.resolution.split(',')))
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w

    inp = "/mnt/HDD4/yiting/liif_modify/load/SCface_SR_MM/probe_d1_crop_ratio_1.5/"
    
    filenames = os.listdir(os.path.join(inp))
    for filename in tqdm(filenames):

        #img = transforms.ToTensor()(Image.open(os.path.join(inp, filename)).convert('RGB'))
        img = transforms.ToTensor()(Image.open(os.path.join(inp, filename)).convert('RGB').resize((32, 32), Image.BICUBIC))
    
        pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
            coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()

        save_path = "/mnt/HDD4/yiting/liif_modify/load/SCface_SR_MM/DIINN_probe_d1_crop_ratio_1.5/"
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        transforms.ToPILImage()(pred).save(os.path.join(save_path, filename.split('.')[0] + '.png'))
    