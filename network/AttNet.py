'''
build model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


from utils.net_utils import convBnRelu, convGnRelu, basicBlock, bottleNeck

class decoderBlock(nn.Module):
    '''
    build decoder block in ETnet
    '''

    def __init__(self, inc, outc, unic, is_up=True, crop=False):
        '''
        #parameters:
        @inc: input channel from previous layer
        @outc: channel of feature maps from encoder layer
        @unic: output channel of decoder block
        @is_up: if upsample input feature maps
        @crop: if crop upsampled feature maps
        '''

        super(decoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inc, outc, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.conv3 = nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outc)
        self.conv4 = nn.Conv2d(outc, unic, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(unic)

        self.relu = nn.ReLU()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.is_up = is_up
        self.crop = crop

    def forward(self, x1, x2):
        '''
        #parameters:
        @x1: input from previous layer
        @x2: input from encoder layer
        '''

        x1 = self.relu(self.bn1(self.conv1(x1)))
        
        if self.is_up == True:
            x1 = self.up(x1)

        if self.crop == True:
            x1 = x1[:,:,:-1,:-1]

        out = x1+x2
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))

        return out

class WB(nn.Module):
    '''
    a weight block
    '''
    
    def __init__(self, inc, outc, reduction_ratio=8):
        super(WB, self).__init__()

        self.basic_conv1 = convBnRelu(inc, outc, 3, p=1)
        self.w1= nn.Sequential(
            nn.Conv2d(outc, outc//reduction_ratio, 1),
            nn.ReLU()
            )
        self.w2 = nn.Sequential(
            nn.Conv2d(outc//reduction_ratio, outc, 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.basic_conv1(x)

        attention1 = self.w2(self.w1(self.avgpool(out)))     
        attention2 = self.w2(self.w1(self.maxpool(out)))

        fout = out * self.sigmoid(attention1 + attention2)

        return fout

class AM(nn.Module):
    '''
    aggregation module
    '''

    def __init__(self, inc1, inc2, inc3, is_1000, outc=128):
        super(AM, self).__init__()

        self.weight_block1 = WB(inc1, outc)
        self.weight_block2 = WB(inc2, outc)
        self.weight_block3 = WB(inc3, outc)

        #self.weight_block12 = WB(outc, outc)
        self.conv1x1_1 = convBnRelu(outc, outc, 1)
        self.conv1x1_2 = convBnRelu(outc, outc, 1)

        self.seg_conv3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),     #500 * 500
            nn.Conv2d(outc, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),     #1000 * 1000
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.is_1000 = is_1000

    def forward(self, x1, x2, x3):
        out1 = self.weight_block1(x1)          #128, 64 * 64
        out2 = self.weight_block2(x2)          #128, 128 * 128
        out3 = self.weight_block3(x3)          #128, 256 * 256

        out1 = self.up(out1)
        if self.is_1000:
            out1 = out1[:,:,:-1,:-1]                               #128, 125 * 125
        out12 = out1+out2                                      #128, 125 * 125
        out12 = self.conv1x1_1(out12)

        out123 = self.up(out12) + out3                        #128, 250 * 250
        out123 = self.conv1x1_2(out123)

        seg_out = self.seg_conv3(out123)

        return seg_out


class AttNet(nn.Module):

    def __init__(self, inc, block, device, nums=[2, 2, 2, 2], is_1000=False):

        super(AttNet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(4, inc, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(inc),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.encoder2 = self._make_layer(inc, inc, block, nums[0], downsample=False)
        self.encoder3 = self._make_layer(inc, 2*inc, block, nums[1])
        self.encoder4 = self._make_layer(2*inc, 4*inc, block, nums[2])
        self.encoder5 = self._make_layer(4*inc, 8*inc, block, nums[3], downsample=False, dilation=2)

        self.decoder1 = decoderBlock(8*inc, 4*inc, 4*inc, False)
        if is_1000:
            self.decoder2 = decoderBlock(4*inc, 2*inc, 2*inc, crop=True)
        else:
            self.decoder2 = decoderBlock(4*inc, 2*inc, 2*inc)
        self.decoder3 = decoderBlock(2*inc, inc, inc)

        self.am = AM(4*inc, 2*inc, inc, is_1000)     

    def forward(self, x):
        
        #################################
        #----------Encoder layer--------#
        #################################        
        en1_out = self.encoder1(x)                      #32 * 500, 500

        en2_out = self.encoder2(self.maxpool(en1_out))  #128, 250, 250
        en3_out = self.encoder3(en2_out)                #256, 125, 125
        en4_out = self.encoder4(en3_out)                #512, 63, 63
        en5_out = self.encoder5(en4_out)                #1024, 63, 63

        #################################
        #----------Decoder layer--------#
        #################################            
        de1_out = self.decoder1(en5_out, en4_out)      #512, 63, 63
        de2_out = self.decoder2(de1_out, en3_out)      #256, 125, 125
        de3_out = self.decoder3(de2_out, en2_out)      #128, 250, 250

        #################################
        #------aggregation module-------#
        ################################# 
        seg_out = self.am(de1_out, de2_out, de3_out)  

        return seg_out

    def _make_layer(self, inc, outc, block, nums, downsample=True, dilation=1):
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
            layers.append(block(inc, outc, stride=2, skip=my_downsample))

        else:
            my_downsample = nn.Sequential(
                nn.Conv2d(inc, outc, 1, stride=1, bias=False),
                nn.BatchNorm2d(outc)
            )
            layers.append(block(inc, outc, skip=my_downsample))
        
        for _ in range(nums-1):
            layers.append(block(outc, outc, dilation=dilation))

        return nn.Sequential(*layers)

def build_network(device, inc=32, is_1000=False):
    return AttNet(inc, basicBlock, device, is_1000=is_1000)    