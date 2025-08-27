import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms

#from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as nnf
"""
def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred
"""
#def batched_predict(model, inp, coord, cell, hr_shape, bsize):
def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            #pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :], hr_shape)
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred
    
def batched_predict_uncertainty(model, inp, coord, cell, hr_shape, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred, _ = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :], hr_shape)
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                # batch['hr_shape']
                pred = model(inp, batch['coord'], batch['cell'])
                #pred = model(inp, batch['coord'], batch['cell'], batch['hr_shape'])
        else:
            # batch['hr_shape']
            pred = batched_predict(model, inp,
                batch['coord'], batch['cell'], eval_bsize)
            #pred = batched_predict(model, inp,
                #batch['coord'], batch['cell'], batch['hr_shape'], eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()

def eval_psnr_uncertainty(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                # batch['hr_shape']
                #pred = model(inp, batch['coord'], batch['cell'])
                pred, _ = model(inp, batch['coord'], batch['cell'], batch['hr_shape'])
        else:
            # batch['hr_shape']
            #pred = batched_predict(model, inp,
                #batch['coord'], batch['cell'], eval_bsize)
            pred = batched_predict_uncertainty(model, inp,
                batch['coord'], batch['cell'], batch['hr_shape'], eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()

def eval_psnr_haar(loader, model1, model2, model3, model4, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()

    idwt = DWTInverse(mode= 'zero', wave= 'haar')

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                pred1 = model1(inp, batch['coord'], batch['cell'])    
                pred2 = model2(inp, batch['coord'], batch['cell'])
                pred3 = model3(inp, batch['coord'], batch['cell'])
                pred4 = model4(inp, batch['coord'], batch['cell'])
                pred1 = (pred1 * 0.5 + 0.5).view(-1, round(inp[0].shape[-2]/2), round(inp[0].shape[-1]/2), 3).permute(0, 3, 1, 2)
                pred2 = (pred2 * 0.5 + 0.5).view(-1, round(inp[0].shape[-2]/2), round(inp[0].shape[-1]/2), 3).permute(0, 3, 1, 2).unsqueeze(0) # torch.Size([1, N, 3, 24, 24])
                pred3 = (pred3 * 0.5 + 0.5).view(-1, round(inp[0].shape[-2]/2), round(inp[0].shape[-1]/2), 3).permute(0, 3, 1, 2).unsqueeze(0) # torch.Size([1, N, 3, 24, 24])
                pred4 = (pred4 * 0.5 + 0.5).view(-1, round(inp[0].shape[-2]/2), round(inp[0].shape[-1]/2), 3).permute(0, 3, 1, 2).unsqueeze(0) # torch.Size([1, N, 3, 24, 24])
                Yl = pred1.cpu()
                Yh = torch.cat((pred2, pred3, pred4), 0) # torch.Size([3, N, 3, 24, 24])
                Yh = Yh.permute(1, 2, 0, 3, 4).cpu() # torch.Size([N, 3, 3, 24, 24])
                Yh_list = []
                Yh_list.append(Yh)
                pred = idwt((Yl, Yh_list)).cuda()
                pred = pred.clamp(0, 1)
                #pred = nnf.interpolate(pred, size=(inp[0].shape[-2], inp[0].shape[-1]), mode='bilinear', align_corners=False)

        """        
        else:
            pred = batched_predict(model, inp, batch['coord'], batch['cell'], eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
        """
        res = metric_fn(pred, batch['hr'])
        val_res.add(res.item(), inp.shape[0])
        
        # save one image for checking
        #print(pred[0].shape)
        #transforms.ToPILImage()(pred[0]).save('debug_pred.png')
        #transforms.ToPILImage()(batch['hr'][0]).save('debug_gt.png')
        #print(pred1[0].shape)
        #print(pred2.squeeze(0)[0].shape)
        #transforms.ToPILImage()(pred1[0].squeeze(0)).save("/home/user/DISK2/data/yiting/representation/liif_modify/debug_00019_rep_ll.png")
        #transforms.ToPILImage()(pred2.squeeze(0)[0].squeeze(0)).save("/home/user/DISK2/data/yiting/representation/liif_modify/debug_00019_rep_lh.png")
        #transforms.ToPILImage()(pred3.squeeze(0)[0].squeeze(0)).save("/home/user/DISK2/data/yiting/representation/liif_modify/debug_00019_rep_hl.png")
        #transforms.ToPILImage()(pred4.squeeze(0)[0].squeeze(0)).save("/home/user/DISK2/data/yiting/representation/liif_modify/debug_00019_rep_hh.png")

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()

def eval_psnr_haar_sr(loader, model1, model2, model3, model4, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()

    idwt = DWTInverse(mode= 'zero', wave= 'haar')

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                print(inp.shape)
                pred1 = model1(inp, batch['coord'], batch['cell'])    
                pred2 = model2(inp, batch['coord'], batch['cell'])
                pred3 = model3(inp, batch['coord'], batch['cell'])
                pred4 = model4(inp, batch['coord'], batch['cell'])
                #print(inp[0].shape[-2])
                pred1 = (pred1 * 0.5 + 0.5).view(-1, inp[0].shape[-2], inp[0].shape[-1], 3).permute(0, 3, 1, 2)
                pred2 = (pred2 * 0.5 + 0.5).view(-1, inp[0].shape[-2], inp[0].shape[-1], 3).permute(0, 3, 1, 2).unsqueeze(0) # torch.Size([1, N, 3, 24, 24])
                pred3 = (pred3 * 0.5 + 0.5).view(-1, inp[0].shape[-2], inp[0].shape[-1], 3).permute(0, 3, 1, 2).unsqueeze(0) # torch.Size([1, N, 3, 24, 24])
                pred4 = (pred4 * 0.5 + 0.5).view(-1, inp[0].shape[-2], inp[0].shape[-1], 3).permute(0, 3, 1, 2).unsqueeze(0) # torch.Size([1, N, 3, 24, 24])
                Yl = pred1.cpu()
                Yh = torch.cat((pred2, pred3, pred4), 0) # torch.Size([3, N, 3, 24, 24])
                Yh = Yh.permute(1, 2, 0, 3, 4).cpu() # torch.Size([N, 3, 3, 24, 24])
                Yh_list = []
                Yh_list.append(Yh)
                pred = idwt((Yl, Yh_list)).cuda()
                #pred = pred.clamp(0, 1)
                print(pred.shape)

        """        
        else:
            pred = batched_predict(model, inp, batch['coord'], batch['cell'], eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
        """
        res = metric_fn(pred, batch['hr'])
        val_res.add(res.item(), inp.shape[0])
        
        # save one image for checking
        #print(pred[0].shape)

        transforms.ToPILImage()(pred[0]).save('cell_haar_pred.png')
        transforms.ToPILImage()(batch['hr'][0]).save('cell_haar_gt.png')

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)
    
    
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    
    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)
    print('result: {:.4f}'.format(res))
    
    """
    model1 = models.make(torch.load(args.model)['model1'], load_sd=True).cuda()
    model2 = models.make(torch.load(args.model)['model2'], load_sd=True).cuda()
    model3 = models.make(torch.load(args.model)['model3'], load_sd=True).cuda()
    model4 = models.make(torch.load(args.model)['model4'], load_sd=True).cuda()

    res = eval_psnr_haar(loader, model1, model2, model3, model4,
        data_norm=config['data_norm'],
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'))

    print('result: {:.4f}'.format(res))
    """
    
    """
    model1 = models.make(torch.load(args.model)['model1'], load_sd=True).cuda()
    model2 = models.make(torch.load(args.model)['model2'], load_sd=True).cuda()
    model3 = models.make(torch.load(args.model)['model3'], load_sd=True).cuda()
    model4 = models.make(torch.load(args.model)['model4'], load_sd=True).cuda()

    res = eval_psnr_haar_sr(loader, model1, model2, model3, model4,
        data_norm=config['data_norm'],
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'))

    print('result: {:.4f}'.format(res))
    """