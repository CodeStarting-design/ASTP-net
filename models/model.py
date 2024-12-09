import numpy as np
import os
import sys
os.chdir(sys.path[0])
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from models.SWCA import *

from models.memory_block import memory_block

from models.TNCA import TNCA_Block


class ASTP(nn.Module): # 最终的输出依旧是单张图片的去雾结果
    def __init__(self, in_chans=3, out_chans=4, 
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[16, 16, 16, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1/4, 1/2, 3/4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN],
                 dilations=[
                    [1, 1, 1, 1, 1, 1, 1, 8],
                    [1, 1, 1, 1, 1, 4, 1, 4],
                    [1, 1, 1, 2, 1, 2, 1, 2],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1]
                 ],
                 kernel_sizes=[9,9,9,0,0],
                 frames_num=3):
        super(ASTP, self).__init__()

        # setting
        self.patch_size = 4
        self.mlp_ratios = mlp_ratios

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans*frames_num, embed_dim=embed_dims[0]*frames_num, kernel_size=3)

        # backbone
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0]*frames_num, depth=depths[0],
                                    num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                    norm_layer=norm_layer[0], kernel_size=kernel_sizes[0],dilations=dilations[0],
                                    attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])
        self.TNCAblock1=TNCA_Block(embed_dims[0],1,kernel_size=kernel_sizes[0],dilation=4,frames_num=frames_num,num_heads=num_heads[0])
        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0]*frames_num, embed_dim=embed_dims[1]*frames_num)

        self.skip1 = nn.Conv2d(embed_dims[0]*frames_num, embed_dims[0], 1)

        self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1]*frames_num, depth=depths[1],
                                 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                 norm_layer=norm_layer[1], kernel_size=kernel_sizes[1],dilations=dilations[1],
                                 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])
        self.TNCAblock2=TNCA_Block(embed_dims[1],1,kernel_size=kernel_sizes[1],dilation=2,frames_num=frames_num,num_heads=num_heads[1])
        
        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1]*frames_num, embed_dim=embed_dims[2]*frames_num)

        self.skip2 = nn.Conv2d(embed_dims[1]*frames_num, embed_dims[1], 1)
         
        self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2]*frames_num, depth=depths[2],
                                 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                 norm_layer=norm_layer[2],kernel_size=kernel_sizes[2],dilations=dilations[2],
                                 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])
        
        self.TNCAblock3=TNCA_Block(embed_dims[2],1,kernel_size=kernel_sizes[2],dilation=1,frames_num=frames_num,num_heads=num_heads[2])

        # 定义记忆力模块
        self.memory_block=memory_block(embed_dims[2])
        
        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
                                 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                 norm_layer=norm_layer[3], kernel_size=kernel_sizes[3],dilations=dilations[3],
                                 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])			

        self.layer5 = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
                                    num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
                                    norm_layer=norm_layer[4], kernel_size=kernel_sizes[4],dilations=dilations[4],
                                    attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)


    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        x,_=self.TNCAblock1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        x,_=self.TNCAblock2(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x)
        memout,qloss=self.memory_block(x)

        _,out=self.TNCAblock3(x)
        x=out+memout

        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)
        return x,qloss

    def forward(self, x):
        b, t, c, h, w = x.size() # 此处的输入就是一个5维的数据
        x=x.view(b, -1, h, w)
        x = self.check_image_size(x) # 在初始时进行了一次补全
        _,_,H,W=x.size()
        feat,qloos = self.forward_features(x)
        K, B = torch.split(feat, (1, 3), dim=1)

        x=x.view(b,t,c,H,W)[:,1,:,:,:]
        x=x.view(b,c,H,W)
        x = K * x - B + x
        x = x[:, :, :h, :w]
        return x,qloos

def ASTP_t():
    return ASTP(
		embed_dims=[24, 48, 96, 48, 24],
		mlp_ratios=[2., 4., 4., 2., 2.],
		depths=[4, 4, 4, 2, 2],
		num_heads=[2, 4, 6, 1, 1],
		attn_ratio=[0, 1/2, 1, 0, 0],
		conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
        dilations=[
            [1, 1, 1, 1],
            [1, 1, 1, 4],
            [1, 2, 1, 2],
            [1, 1],
            [1, 1]
         ],
        kernel_sizes=[9,9,9,0,0],
        frames_num=3)


def ASTP_s():
    return ASTP(
		embed_dims=[24, 48, 96, 48, 24],
		mlp_ratios=[2., 4., 4., 2., 2.],
		depths=[8, 8, 8, 4, 4],
		num_heads=[2, 4, 6, 1, 1],
		attn_ratio=[1/4, 1/2, 3/4, 0, 0],
		conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
        dilations=[
            [1, 1, 1, 1, 1, 1, 1, 8],
            [1, 1, 1, 1, 1, 4, 1, 4],
            [1, 1, 1, 2, 1, 2, 1, 2],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
         ],
        kernel_sizes=[13,13,11,0,0],
        frames_num=3)


def ASTP_b():
    return ASTP(
        embed_dims=[24, 48, 96, 48, 24],
		mlp_ratios=[2., 4., 4., 2., 2.],
		depths=[16, 16, 16, 8, 8],
		num_heads=[2, 4, 6, 1, 1],
		attn_ratio=[1/4, 1/2, 3/4, 0, 0],
		conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
        dilations=[
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 8],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4, 1, 4, 1, 4],
            [1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
         ],
        kernel_sizes=[9,9,9,0,0],
        frames_num=3)

