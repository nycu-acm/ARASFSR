import numpy as np
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity
from skimage.metrics import normalized_root_mse
from PIL import Image
import os
from tqdm import tqdm
import lpips
import torch
import pandas as pd

class util_of_lpips():
    def __init__(self, net, use_gpu=True):
        '''
        Parameters
        ----------
        net: str, ['alex', 'vgg']
        use_gpu: bool
        Returns
        -------
        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        ## Initializing the model
        self.loss_fn = lpips.LPIPS(net=net)
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    def calc_lpips(self, img1_path, img2_path):
        '''
        Parameters
        ----------
        img1_path : str
        img2_path : str
        Returns
        -------
        dist01 : torch.Tensor

        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        # Load images
        img0 = lpips.im2tensor(lpips.load_image(img1_path))  # RGB image from [-1,1]
        img1 = lpips.im2tensor(lpips.load_image(img2_path))

        if self.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()
        with torch.no_grad():
            dist01 = self.loss_fn.forward(img0, img1)
        return dist01.item()
        
def rgb2y_matlab(x):
    """Convert RGB image to illumination Y in Ycbcr space in matlab way.
    -------------
    # Args
        - Input: x, byte RGB image, value range [0, 255]
        - Ouput: byte gray image, value range [16, 235] 

    # Shape
        - Input: (H, W, C)
        - Output: (H, W) 
    """
    K = np.array([65.481, 128.553, 24.966]) / 255.0
    Y = 16 + np.matmul(x, K)
    return Y.astype(np.uint8)


def PSNR(im1, im2, use_y_channel=True):
    """Calculate PSNR score between im1 and im2
    --------------
    # Args
        - im1, im2: input byte RGB image, value range [0, 255]
        - use_y_channel: if convert im1 and im2 to illumination channel first
    """
    if use_y_channel:
        im1 = rgb2y_matlab(im1)
        im2 = rgb2y_matlab(im2)
    im1 = im1.astype(np.float)
    im2 = im2.astype(np.float)
    mse = np.mean(np.square(im1 - im2)) 
    return 10 * np.log10(255**2 / mse) 


def SSIM(gt_img, noise_img):
    """Calculate SSIM score between im1 and im2 in Y space
    -------------
    # Args
        - gt_img: ground truth image, byte RGB image
        - noise_img: image with noise, byte RGB image
    """
    gt_img = rgb2y_matlab(gt_img)
    noise_img = rgb2y_matlab(noise_img)
     
    #ssim_score = compare_ssim(gt_img, noise_img, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    ssim_score = structural_similarity(gt_img, noise_img)
    return ssim_score
    
def MPS(gt_path, test_path, lpips_net):
    '''
        Mean Perceptual Score, From AIM 2020: Scene Relighting and Illumination Estimation Challenge.
    '''
    ssim = SSIM(np.array(Image.open(gt_path)), np.array(Image.open(test_path)))
    lpips = lpips_net.calc_lpips(gt_path, test_path)
    mps = 0.5 * (ssim + (1 - lpips))
    return mps
    
def NRMSE(gt_path, test_path, len_dir):
    '''
        Use Openface landmark result to calculate NRMSE of GT landmark and Test landmark
    '''
    img_anno_list_gt = []
    for i in range(len_dir):
        temp = []
        img_anno_list_gt.append(temp)

    df_gt = pd.read_csv(gt_path)
    for i in range(68):   
        for index, (anno_x, anno_y) in enumerate(zip(df_gt[f' x_{i}'], df_gt[f' y_{i}'])):
            img_anno_list_gt[index].append([anno_x, anno_y]) 
    
    img_anno_list_test = []
    for i in range(len_dir):
        temp = []
        img_anno_list_test.append(temp)      
    df_test = pd.read_csv(test_path)      
    for i in range(68):
        for index, (anno_x, anno_y) in enumerate(zip(df_test[f' x_{i}'], df_test[f' y_{i}'])):
            img_anno_list_test[index].append([anno_x, anno_y])
    
    nrmse = 0
    for index in range(len(img_anno_list_gt)):
        nrmse_image = 0
        for i in range(68):
            nrmse_image += np.linalg.norm(np.array(img_anno_list_gt[index][i]) - np.array(img_anno_list_test[index][i]))
        nrmse_image /= 68
        nrmse += nrmse_image
        
    return nrmse   
    

def psnr_ssim_lpips_dir(gt_dir, test_dir, lpips_net, gt_csv='', test_csv=''):
    gt_img_list = sorted([x for x in sorted(os.listdir(gt_dir))])
    test_img_list = sorted([x for x in sorted(os.listdir(test_dir))])
   
    #  assert gt_img_list == test_img_list, 'Test image names are different from gt images.' 
    
    #gt_csv = "/mnt/HDD4/yuwei/dataset/pkl/gt_csv/512_bic_test.csv"
    gt_csv = "/mnt/HDD4/yuwei/dataset/pkl/gt_csv/Helen_HR.csv"
    test_csv = "/mnt/HDD4/yuwei/dataset/pkl/test_csv/FSRNet/results_helen.csv"
    
    psnr_score = 0
    ssim_score = 0
    lpips_score = 0
    mps_score = 0
    nrmse_score = 0
    
    for gt_name, test_name in tqdm(zip(gt_img_list, test_img_list), total=len(gt_img_list)):
        gt_img = Image.open(os.path.join(gt_dir, gt_name))
        test_img = Image.open(os.path.join(test_dir, test_name))
        gt_img = np.array(gt_img)
        test_img = np.array(test_img)
        psnr_score += PSNR(gt_img, test_img)
        ssim_score += SSIM(gt_img, test_img)
        lpips_score += lpips_net.calc_lpips(os.path.join(gt_dir, gt_name), os.path.join(test_dir, test_name))
        mps_score += MPS(os.path.join(gt_dir, gt_name), os.path.join(test_dir, test_name), lpips_net)
    
    nrmse_score = NRMSE(gt_csv, test_csv, len(gt_img_list))
    
    return psnr_score / len(gt_img_list), ssim_score / len(gt_img_list), lpips_score / len(gt_img_list), mps_score / len(gt_img_list), nrmse_score / len(gt_img_list)

if __name__ == '__main__':
    #gt_dir = "/mnt/HDD4/yuwei/dataset/512_bic_test/"
    #gt_dir = "/mnt/HDD4/yuwei/dataset/256_bic_test/"
    #gt_dir = "/mnt/HDD4/yuwei/dataset/128_bic_test/"
    gt_dir = "/mnt/HDD4/yuwei/dataset/Helen_test/HR/" 
    test_dirs = [
            "/mnt/HDD4/yuwei/FSRNet-pytorch/results/Helen/"
            ]
    lpips_net = util_of_lpips('vgg')
    for td in test_dirs:
        result = psnr_ssim_lpips_dir(td, gt_dir, lpips_net)
        print(td, result)



