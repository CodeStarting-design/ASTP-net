import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
import time

os.chdir(sys.path[0])
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from utils import AverageMeter
from dataset.dataloader import DehazeDataset
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ASTP-s', type=str, help='model name')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='HazeWorld', type=str, help='dataset name')
parser.add_argument('--testdataset', default='HazeWorld', type=str, help='test dataset name')
parser.add_argument('--exp', default='outdoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



def train(train_loader, network, criterion, optimizer, scaler,epoch,changeEpoch=30):
    losses = AverageMeter()

    torch.cuda.empty_cache()
    
    network.train()

    for batch in train_loader:
        inputs, targets = batch
        b,_,c,h,w=targets.size()
        target=targets[:,1,:,:,:].view(b,c,h,w).cuda()
        inputs=inputs.cuda()
        with autocast(args.no_autocast):
            output,qloss = network(inputs)
            L1_loss = criterion(output,target)
        losses.update(L1_loss.item())
        if epoch<=changeEpoch:
            loss=L1_loss*0.95+qloss*0.05
        else:
            loss=L1_loss
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return losses.avg


def valid(val_loader, network):
    PSNR = AverageMeter()
    losses = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader:
        inputs, targets = batch
        b,_,c,h,w=targets.size()
        target=targets[:,1,:,:,:].view(b,c,h,w).cuda()
        inputs=inputs.cuda()

        with torch.no_grad():							# torch.no_grad() may cause warning
            output,_ = network(inputs)
        output=output.clamp_(-1, 1)		

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), inputs.size(0))
        losses.update(F.mse_loss(output * 0.5 + 0.5, target * 0.5 + 0.5).item())
        

    return PSNR.avg,losses.avg


if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model+'.json')  
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network).cuda()

    criterion = nn.L1Loss()

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer") 

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    testdataset_dir = os.path.join(args.data_dir, args.testdataset)
    

    train_dataset = DehazeDataset(dataset_dir,'train', 'train',  
                              setting['width'],setting['height'])
    train_loader = DataLoader(train_dataset, 
                            batch_size=setting['batch_size'], 
                            shuffle=True,
                            num_workers=3,
                            pin_memory=True,
                            drop_last=True)
    val_dataset = DehazeDataset(testdataset_dir, 'test', setting['valid_mode'], 
                              setting['width'],setting['height'])
    val_loader = DataLoader(val_dataset,
                            batch_size=40,
                            num_workers=1,
                            pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(os.path.join(save_dir, args.model+'.pth')): # 不存在预训练模型
        print('==> Start training, current model name: ' + args.model)
        # print(network)

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

        best_psnr = 0
        for epoch in tqdm(range(setting['epochs'] + 1)):
            loss = train(train_loader, network, criterion, optimizer, scaler,epoch,setting['changeEpoch'])

            writer.add_scalar('train_loss', loss, epoch)

            scheduler.step()

            if epoch % setting['eval_freq'] == 0:
                avg_psnr,val_loss = valid(val_loader, network)
                
                writer.add_scalar('valid_psnr', avg_psnr, epoch)
                writer.add_scalar('valid_loss', val_loss, epoch)

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict(),
                                'epoch': epoch,
                                'best_psnr':best_psnr,
                                'optimizer': optimizer.state_dict(),
                                'lr_schedule': scheduler.state_dict()},
                               os.path.join(save_dir, args.model+'.pth'))
                
                writer.add_scalar('best_psnr', best_psnr, epoch)
 
    else:  # 存在预训练模型，从预训练模型中开始训练
        checkpoint_path = os.path.join(save_dir, args.model+'.pth')
        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_psnr = checkpoint['best_psnr']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_schedule'])
        print('==> Recover training, current model name: ' + args.model)
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

        
        for epoch in tqdm(range(start_epoch+1 ,setting['epochs'] + 1)):
            loss = train(train_loader, network, criterion, optimizer, scaler,epoch)

            writer.add_scalar('train_loss', loss, epoch)

            scheduler.step()

            if epoch % setting['eval_freq'] == 0:
                avg_psnr,val_loss = valid(val_loader, network)
                
                writer.add_scalar('valid_psnr', avg_psnr, epoch)
                writer.add_scalar('valid_loss', val_loss, epoch)

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'state_dict': network.state_dict(),
                                'epoch': epoch,
                                'best_psnr':best_psnr,
                                'optimizer': optimizer.state_dict(),
                                'lr_schedule': scheduler.state_dict()},
                               os.path.join(save_dir, args.model+'.pth'))
                
                writer.add_scalar('best_psnr', best_psnr, epoch)
        
