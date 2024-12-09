import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
import sys

os.chdir(sys.path[0])
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from dataset.dataloader import DehazeDataset
from models import *

from utils import AverageMeter, write_img, chw_to_hwc
import time


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ASTP-s', type=str, help='model name')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='HazeWorld', type=str, help='dataset name')
parser.add_argument('--exp', default='outdoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def single(save_dir):
    state_dict = torch.load(save_dir)['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict


def test(test_loader, network, result_dir):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    os.makedirs(result_dir, exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in enumerate(test_loader):
        torch.cuda.synchronize()
        # start = time.time()

        inputs, targets, res_dir = batch
        b,_,c,h,w=targets.size()
        target=targets[:,1,:,:,:].view(b,c,h,w).cuda()
        target=target.half()
        inputs=inputs.cuda()
        inputs=inputs.half()

        with torch.no_grad():
            output,_ = network(inputs)
            output=output.clamp_(-1, 1)	

            # [-1, 1] to [0, 1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))		
            ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))), 
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False).item()				

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)
        torch.cuda.synchronize()
        # end = time.time()
        print('Test: [{0}]\t'
              'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
              'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
              .format(idx, psnr=PSNR, ssim=SSIM))
        
        # elapsed_time = end - start

        # print("time: {:.4f} s".format(elapsed_time))

        f_result.write('%s,%.02f,%.03f\n'%(idx, psnr_val, ssim_val))

        # out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        # write_img(os.path.join(result_dir, res_dir[0]), out_img)

    f_result.close()

    os.rename(os.path.join(result_dir, 'results.csv'), 
              os.path.join(result_dir, '%.02f | %.04f.csv'%(PSNR.avg, SSIM.avg)))


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model+'.json')  
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)
    network = eval(args.model.replace('-', '_'))()
    network.half()
    network = nn.DataParallel(network).cuda()
    saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'.pth')

    if os.path.exists(saved_model_dir):
        print('==> Start testing, current model name: ' + args.model)
        checkpoint = torch.load(saved_model_dir)
        network.load_state_dict(checkpoint['state_dict'])
    else:
        print('==> No existing trained model!')
        exit(0)

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    test_dataset = DehazeDataset(dataset_dir,'test', 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             pin_memory=True)

    result_dir = os.path.join(args.result_dir, args.dataset, args.model)
    test(test_loader, network, result_dir)