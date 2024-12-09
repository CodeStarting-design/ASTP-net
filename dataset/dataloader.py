import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import cv2
import os
import sys
os.chdir(sys.path[0])
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from utils import hwc_to_chw, read_img

def augment(imgs=[], width=720,height=720, edge_decay=0., only_h_flip=False): # 要对连续三帧进行处理
    _ , _ , H, W = imgs[0].shape
    Hc, Wc = [height,width]

    # simple re-weight for the edge
    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H-Hc)

    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W-Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][:,:,Hs:(Hs+Hc), Ws:(Ws+Wc)]

    # horizontal flip
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = torch.flip(imgs[i], [3])

    if not only_h_flip:
        # bad data augmentations for outdoor
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = torch.rot90(imgs[i], rot_deg, (2, 3))
            
    return imgs


def align(imgs=[], width=500,height=500):
    _ , _ , H, W = imgs[0].shape
    Hc, Wc = [height,width]

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][:,:,Hs:(Hs+Hc), Ws:(Ws+Wc)]

    return imgs

class DehazeDataset(Dataset):
    def __init__(self, data_dir,sub_dir,mode,width=500,height=500, edge_decay=0, only_h_flip=False): # 每次需要取出的是三帧，在数据集读取时将连续三帧的文件路径直接保存
        
        self.root_dir = os.path.join(data_dir,sub_dir)
        self.width = width
        self.height = height 
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip
        self.mode=mode

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.input_frames = [] # 保存相对路径
        self.target_frames = []
        self.scenes = sorted(os.listdir(os.path.join(self.root_dir, 'hazy'))) # 场景文件夹的名称
        for i in range(len(self.scenes)):
            scene_folder = self.scenes[i]
            hazy_scene_path = os.path.join(self.root_dir, 'hazy', scene_folder)
            gt_scene_path = os.path.join(self.root_dir, 'gt', scene_folder)
            video_folders = sorted(os.listdir(hazy_scene_path)) # 场景下的全部雾霾视视频文件夹名
            for video_folder in video_folders: # 遍历某一个文件夹
                hazy_video_path = os.path.join(hazy_scene_path, video_folder) # 某一个文件夹的名字
                gt_video_path = os.path.join(gt_scene_path, video_folder)
                frame_files_hazy = sorted(os.listdir(hazy_video_path)) # 在文件夹中的每一帧
                frame_files_gt = sorted(os.listdir(gt_video_path))
                for j in range(len(frame_files_hazy)-2):
                    frame_hazy_path=[]
                    frame_gt_path=[]
                    for k in range(3):
                        try:
                            frame_hazy_path.append(os.path.join(hazy_video_path,frame_files_hazy[j+k]))
                            frame_gt_path.append(os.path.join(gt_video_path,frame_files_gt[j+k]))
                        except:
                            print(gt_video_path)
                    self.input_frames.append(frame_hazy_path)
                    self.target_frames.append(frame_gt_path)
        

    def __len__(self):
        return len(self.input_frames)

    def __getitem__(self, idx):
        """
        在该方法中，首先根据文件路径取出相应的连续三帧
        再对这三帧进行简单的预处理
        沿着第一维度进行堆叠，返回
        """
        input_frames = []
        target_frames = []
        input_frames_path=self.input_frames[idx] # 取出的是文件路径
        target_frames_path=self.target_frames[idx]
        for i in range(len(input_frames_path)):
            input_frame_path=input_frames_path[i]
            target_frame_path=target_frames_path[i]

            frame_hazy = Image.open(input_frame_path).convert('RGB')
            frame_hazy = self.transform(frame_hazy)
            frame_hazy=(frame_hazy)*2-1
            
            frame_gt = Image.open(target_frame_path).convert('RGB')
            frame_gt = self.transform(frame_gt)
            frame_gt=(frame_gt)*2-1

            input_frames.append(frame_hazy)
            target_frames.append(frame_gt)

        # 最高高维堆叠
        input_frames = torch.stack(input_frames, dim=0)
        target_frames = torch.stack(target_frames, dim=0)

        # 使用os.path模块获取文件路径的各个部分
        dir_path, file_name = os.path.split(input_frame_path)
        _, last_dir = os.path.split(dir_path)
        _, second_last_dir = os.path.split(_)

        # 构建后三层文件路径字符串
        result_path = os.path.join(second_last_dir, last_dir, file_name)

        if self.mode == 'train':
            # 进行数据增强
            [input_frames,target_frames]=augment([input_frames,target_frames],self.width,self.height,self.edge_decay,self.only_h_flip)
        
        if self.mode == 'valid':
            [input_frames, target_frames] = align([input_frames, target_frames], self.width,self.height)
        if self.mode == 'test':
            # _,_,height,_=input_frames.size()
            # [input_frames, target_frames] = align([input_frames, target_frames], 720,height)
            return input_frames,target_frames,result_path
        return input_frames,target_frames
        

# train_dataset = DehazeDataset('../data/HazeWorld/','train', 'train')
# print(len(train_dataset))
# train_loader = DataLoader(train_dataset, 
#                             batch_size=1, 
#                             shuffle=False,
#                             drop_last=True)
                            
