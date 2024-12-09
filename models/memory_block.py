import torch
from torch import nn
import torch.nn.functional as F  
import numpy as np
from einops import rearrange

class memory_block(nn.Module): # 
    def __init__(self, in_channels=128):
        super(memory_block, self).__init__()
        self.in_channels = in_channels 
        self.inter_channels = self.in_channels // 2
                
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1) 

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=self.in_channels, eps=1e-6, affine=True) 
        
        self.W_z1 = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
        nn.init.constant_(self.W_z1.weight, 0)
        nn.init.constant_(self.W_z1.bias, 0)
        self.mb = torch.nn.Parameter(torch.randn(self.inter_channels, 256)) 
    
    def forward(self, x):
        b,c,h,w=x.size()
        
        x=x.view(b,3, c//3, h, w).contiguous()
        q = x[:, 1, :, :, :] # 第2帧也就是当前帧

        q_=self.norm(q) 
        
        phi_x = self.phi(q_).view(b,self.inter_channels, -1) 
        phi_x_for_quant=phi_x.permute(0,2,1) 
        
        mbg = self.mb.unsqueeze(0).repeat(b, 1, 1) 
        f1 = torch.matmul(phi_x_for_quant, mbg) 
        f_div_C1 = F.softmax(f1 * (int(self.inter_channels) ** (-0.5)), dim=-1) 
        y1 = torch.matmul(f_div_C1, mbg.permute(0, 2, 1)) 
        qloss=torch.mean(torch.abs(phi_x_for_quant-y1)) 
        y1 = y1.permute(0, 2, 1).view(b, self.inter_channels, h, w).contiguous() 
        W_y1 = self.W_z1(y1) 
        
        z =q+W_y1 

        return z, qloss

