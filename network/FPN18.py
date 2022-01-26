'''
build FPN with ResNet18
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.net_utils import convBnRelu, basicBlock

class LC(nn.Module):
    '''
    Lateral Connection

    #Arguments:
    @inc: channel of x2
    @is_up: if upsampling x1
    @is_crop: if crop x1 after upsampling
    '''

    def __init__(self, inc, is_up=True, is_crop=None):
        super(LC, self).__init__()
        self.skip_connection = convBnRelu(inc, 256, 1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.is_up = is_up
        self.is_crop = is_crop

    def forward(self, x1, x2):
        '''
        x1: from previous layer
        x2: from encoder,
        '''
        if self.is_up:
            x1 = self.up(x1)

        if self.is_crop is not None:
            x1 = x1[:,:,:-self.is_crop,:-self.is_crop]

        x2 = self.skip_connection(x2)
        out = x1 + x2

        return out

class FPN18(nn.Module):
    
    def __init__(self, imgc=4, inc=32, nums=[2, 2, 2, 2]):
        super(FPN18, self).__init__()
        
        self.encoder1 = convBnRelu(imgc, inc, 7, 2, 3)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoder2 = self._make_layer(inc, inc, nums[0], downsample=False)
        self.encoder3 = self._make_layer(inc, 2*inc, nums[1])
        self.encoder4 = self._make_layer(2*inc, 4*inc, nums[2])
        self.encoder5 = self._make_layer(4*inc, 8*inc, nums[3], downsample=False, dilation=2)

        self.lc5 = convBnRelu(8*inc, 256, 1)
        self.lc4 = LC(4*inc, False)
        self.lc3 = LC(2*inc)
        self.lc2 = LC(inc)
        
        
        f_blocks = []
        for i in range(4):
                        f_blocks.append(nn.Sequential(
                        convBnRelu(256, 128, 3, 1, 1),
                        convBnRelu(128, 128, 3, 1, 1))
                                )
        self.f_blocks = nn.ModuleList(f_blocks)
        
        self.up5 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.agg = convBnRelu(512, 128, 3, 1, 1)      

    def forward(self, x):        
        #################################
        #------------bottom up----------#
        #################################           

        en1_out = self.encoder1(x)                              #1/2 * 1/2
        en2_out = self.encoder2(self.maxpool(en1_out))          #1/4 * 1/4
        en3_out = self.encoder3(en2_out)                        #1/8 * 1/8
        en4_out = self.encoder4(en3_out)                        #1/16 * 1/16
        en5_out = self.encoder5(en4_out)                        #1/16 * 1/16

        #################################
        #------------top down-----------#
        #################################       

        lc5_out = self.lc5(en5_out)                     
        lc4_out = self.lc4(lc5_out, en4_out)
        lc3_out = self.lc3(lc4_out, en3_out)
        lc2_out = self.lc2(lc3_out, en2_out)

        #################################
        #----------aggregation----------#
        #################################      
        stage5 = self.up5(self.f_blocks[0](lc5_out))             #1/4 * 1/4
        stage4 = self.up4(self.f_blocks[1](lc4_out))             #1/4 * 1/4
        stage3 = self.up3(self.f_blocks[2](lc3_out))             #1/4 * 1/4
        stage2 = self.f_blocks[3]((lc2_out))                     #1/4 * 1/4
        
        out = torch.cat([stage5, stage4, stage3, stage2], dim=1)
        out = self.agg(out)                                     #1/4 * 1/4 * 128

        return out

    def _make_layer(self, inc, outc, nums, downsample=True, dilation=1):
        '''
        #inc: input channel from previous layer
        #outc: output channel of this block
        '''
        layers = []

        if downsample == True:
            my_downsample = nn.Sequential(
                nn.Conv2d(inc, outc, 1, stride=2, bias=False),
                nn.BatchNorm2d(outc)
            )
            layers.append(basicBlock(inc, outc, stride=2, skip=my_downsample))

        else:
            my_downsample = nn.Sequential(
                nn.Conv2d(inc, outc, 1, stride=1, bias=False),
                nn.BatchNorm2d(outc)
            )
            layers.append(basicBlock(inc, outc, skip=my_downsample))
        
        for _ in range(nums-1):
            layers.append(basicBlock(outc, outc, dilation=dilation))

        return nn.Sequential(*layers)
