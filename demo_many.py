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
python demo_many.py --model "/mnt/HDD4/yiting/liif_modify/save/_final_train_celebAHQ-64-128_liif_save/epoch-best.pth" --method_backbone liif_edsr --resolution_in 64,64 --resolution_out 128,128 --downsampling bic --gpu 1


"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--method_backbone')
    parser.add_argument('--resolution_in')
    parser.add_argument('--resolution_out')
    parser.add_argument('--downsampling')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    
    
    # file(file)
    h, w = list(map(int, args.resolution_out.split(',')))
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    
    l_h, l_w = list(map(int, args.resolution_in.split(',')))
    
    in_p = '/mnt/HDD4/yiting/liif_modify/load/celebAHQ/'+str(l_h)+'_'+args.downsampling
    out_p = os.path.join('/mnt/HDD4/yiting/liif_modify/load/celebAHQ_SR_MM/'+args.method_backbone, str(l_h)+'_'+args.downsampling, str(h))
    os.mkdir(out_p)
    filenames = os.listdir(in_p)
    for filename in tqdm(filenames):

        img = transforms.ToTensor()(Image.open(os.path.join(in_p, filename)).convert('RGB'))
    
        pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
            coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
        pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()

        transforms.ToPILImage()(pred).save(os.path.join(out_p, filename.split('.')[0] + '.png'))
    