import os
import numpy as np
from PIL import Image, ImageChops 
from torchvision import transforms
import torch

def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)

img1 = Image.open("/home/user/DISK2/data/yiting/representation/liif_modify/101087x2.png").convert("L")
img2= Image.open("/home/user/DISK2/data/yiting/representation/liif_modify/101087x4_trainx4_testx2.png").convert("L")
#img1 = Image.open("/home/user/DISK2/data/yiting/representation/liif_modify/101087x4_trainx1-x4_testx2.png").convert('RGB')
#img1 = Image.open("/home/user/DISK2/data/yiting/representation/liif_modify/101087x4_trainx4_testx2.png").convert('RGB')
#img2 = Image.open("/home/user/DISK2/data/yiting/representation/liif_modify/101087x2.png").convert('RGB')
#print(calc_psnr(transforms.ToTensor()(img2), transforms.ToTensor()(img1)))

print(np.array(img1)- np.array(img2))
img3 = ImageChops.subtract(img1, img2)
img = np.array(img3)
minval = img.min()
maxval = img.max()
img = img*(255.0/(maxval-minval)) 
new_img = Image.fromarray(img.astype('uint8'))
new_img.save('diff.png')

'''
def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def demo_normalize():
    img = Image.open('101087_output_x8_train.png').convert('RGBA')
    arr = np.array(img)
    new_img = Image.fromarray(normalize(arr).astype('uint8'),'RGBA')
    new_img.save('/tmp/normalized.png')
'''