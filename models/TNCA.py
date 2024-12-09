import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
from torch.nn.functional import pad
from natten.functional import natten2dqkrpb, natten2dav

class TimeNeighborhoodAttention(nn.Module):
    """
    Time Neighborhood Attention 2D Module
    dim:单帧的维度
    frames_num:帧数
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, 
                 dilation=None, frames_num=3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        if type(dilation) is str:
            self.dilation = None
            self.window_size = None
        else:
            assert dilation is None or dilation >= 1, \
                f"Dilation must be greater than or equal to 1, got {dilation}."
            self.dilation = dilation or 1
            self.window_size = self.kernel_size * self.dilation
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))) # 
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.frames_num=frames_num
        self.dim=dim
        self.W_z = nn.Conv2d(dim*frames_num, dim, kernel_size=1)

    def forward(self, q, k, v):# 此处传入的就是qkv
        B, C, Hp, Wp = q.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        dilation = self.dilation
        window_size = self.window_size
        if window_size is None:
            dilation = max(min(H, W) // self.kernel_size, 1)
            window_size = dilation * self.kernel_size
        if H < window_size or W < window_size: # 在去雾任务中，实际上并不会导致出现填充问题
            pad_l = pad_t = 0
            pad_r = max(0, window_size - W)
            pad_b = max(0, window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b),"reflect")
            _, _, H, W, = x.shape

        q_t=q.reshape(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2)
        k_n=k.reshape(B, self.frames_num, self.dim, H, W)
        v_n=v.reshape(B, self.frames_num, self.dim, H, W)
        q_t=q_t * self.scale

        x_n=[]
        
        for i in range(self.frames_num):
            k_t=k_n[:,i,:,:,:].reshape(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2)
            v_t=v_n[:,i,:,:,:].reshape(B, self.num_heads, self.head_dim, H, W).permute(0, 1, 3, 4, 2)
            attn=natten2dqkrpb(q_t,k_t,self.rpb,self.kernel_size,dilation)
            attn = attn.softmax(dim=-1)
            x = natten2dav(attn, v_t, self.kernel_size,dilation)
            x = x.permute(0, 1, 4, 2, 3).reshape(B, C, H, W)
            x_n.append(x)
        x_n=torch.cat(x_n, dim=1)
        x_t=self.W_z(x_n)
        return x_t
    

class TNCA_Block(nn.Module):
    """传递的参数:
    dim:单帧视频对应的通道数,
    ratio:qk通道压缩的比率,
    frames_num:视频帧的数量
    """
    def __init__(self,dim,ratio, kernel_size, dilation, 
                 frames_num=3,num_heads=2, qkv_bias=False, qk_scale=None): 
        super(TNCA_Block, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        qkplanes=int(dim * ratio)
        self.frames_num=frames_num
        self.qkplanes=qkplanes
        self.q = nn.Conv2d(dim,dim,1)
        self.k = nn.Conv2d(dim*frames_num,dim*frames_num,1)
        self.v = nn.Conv2d(dim*frames_num, dim*frames_num,1)
        self.proj = nn.Conv2d(dim*frames_num, dim,1)
        self.norm = torch.nn.GroupNorm(num_groups=dim//12, num_channels=dim, eps=1e-6, affine=True) # 定义组归一化
        self.TimeAtten=TimeNeighborhoodAttention(dim,kernel_size,num_heads,dilation=dilation,frames_num=frames_num)
        
    def forward(self, x):# 此处传入的是相邻帧特征
        # 先对相邻帧特征进行处理
        b,c,h,w=x.size()
        x_n=x
        x=x.view(b, 3, c//3, h, w)
        x_t = x[:, 1, :, :, :] # 取出第2帧也就是当前帧
        x_t=self.norm(x_t) # 维度是B，C，H，W
        x_n=x_n.view(b*3,c//3,h,w)
        x_n=self.norm(x_n)
        x_n=x_n.view(b, c, h, w).contiguous()

        # 选取当前帧作为query
        q = self.q(x_t)
        k = self.k(x_n)
        v = self.v(x_n)

        x_t=self.TimeAtten(q,k,v)
        return x_n,x_t # 返回的x_n和x_t都是b,c,h,w维度的
