'''
build model
''' 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class basicBlock(nn.Module):
    '''
    build basic block in residual net, 2 layers
    '''
    
    def __init__(self, inc, outc, stride=1, dilation=1, skip=None):
        super(basicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inc, outc, 3, stride=stride, \
                               dilation=dilation, padding=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc, outc, 3, dilation=dilation,
                                padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU()

        self.skip = skip

    def forward(self, x):
        shortcut = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.skip is not None:
            shortcut = self.skip(x)

        out += shortcut
        out = self.relu(out)

        return out

class bottleNeck(nn.Module):
    '''
    build bottle_neck in residual net, 3 layers
    '''

    def __init__(self, inc, outc, stride=1, dilation=1, skip=None):
        '''
        #parameters:
        @inc: input channel from layer
        @outc: output channel //4
        '''
        super(bottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inc, outc, stride=stride, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1, bias=False)  
        self.bn2 = nn.BatchNorm2d(outc)
        self.conv3 = nn.Conv2d(outc, outc*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outc*4)
        self.relu = nn.ReLU()
        self.skip = skip

    def forward(self, x):
        
        shortcut = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            shortcut = self.skip(x)

        out += shortcut
        out = self.relu(out)

        return out


class convBnRelu(nn.Module):
    '''
    conv2d -> batch normalizatoin -> relu
    '''

    def __init__(self, inc, outc, k, s=1, p=0):
        super(convBnRelu, self).__init__()
        self.module = nn.Sequential(
                    nn.Conv2d(inc, outc, k, s, padding=p, bias=False),
                    nn.BatchNorm2d(outc),
                    nn.ReLU())
    
    def forward(self, x):

        return self.module(x)


class lc(nn.Module):

    def __init__(self, inc, is_up=True, is_crop=False):
        super(lc, self).__init__()
        self.skip_connection = convBnRelu(inc, 256, 1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.is_up = is_up
        self.is_crop = is_crop

    def forward(self, x1, x2):
        '''
        x1: from encoder,
        x2: from previous layer
        '''

        x1 = self.skip_connection(x1)

        if self.is_up:
            x2 = self.up(x2)

        if self.is_crop:
            x2 = x2[:,:,:-1,:-1]

        out = x1 + x2

        return out

class edgeNet(nn.Module):
    def __init__(self):
        super(edgeNet, self).__init__()
        self.conv1 = convBnRelu(3, 64, 3, p=1)
        self.conv2 = convBnRelu(64, 64, 3, p=1)
        self.conv3 = convBnRelu(64, 64, 3, p=1)
        self.conv4 = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out

class fpn18(nn.Module):
    def __init__(self, is_1000, inc=32, nums=[2, 2, 2, 2]):
        super(fpn18, self).__init__()
        self.en1 = nn.Sequential(
            nn.Conv2d(3, inc, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(inc),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.en2 = self._make_layer(nums[0], inc, inc, False)
        self.en3 = self._make_layer(nums[1], inc, inc*2)
        self.en4 = self._make_layer(nums[2], inc*2, inc*4)
        self.en5 = self._make_layer(nums[3], inc*4, inc*8)

        self.lc5 = convBnRelu(inc*8, 256, 1)
        
        if is_1000 == True:
            self.lc4 = lc(inc*4, True, True)
            self.lc3 = lc(inc*2, True, True)
        else:
            self.lc4 = lc(inc*4, True, False)
            self.lc3 = lc(inc*2, True, False)
        
        self.lc2 = lc(inc, True, False)

        self.f5 = nn.Sequential(
            convBnRelu(256, 128, 1, 1),
            convBnRelu(128, 128, 3, 1, 1)
        )
        self.f4 = nn.Sequential(
            convBnRelu(256, 128, 1, 1),
            convBnRelu(128, 128, 3, 1, 1)
        )
        self.f3 = nn.Sequential(
            convBnRelu(256, 128, 1, 1),
            convBnRelu(128, 128, 3, 1, 1)
        )
        self.f2 = nn.Sequential(
            convBnRelu(256, 128, 1, 1),
            convBnRelu(128, 128, 3, 1, 1)
        )

        self.up5 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.recovery = nn.Sequential(
            convBnRelu(512, 128, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            convBnRelu(128, 64, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 1, 3, 1, padding=1),
            nn.Sigmoid()
        )
        
        self.is_1000 = is_1000

    def forward(self, x):
        #################################
        #------------bottom up----------#
        #################################            
        en1_out = self.en1(x)                         #500, 64*64
 
        en2_out = self.en2(self.maxpool(en1_out))     #250, 64*64    
        en3_out = self.en3(en2_out)                   #125, 128*128
        en4_out = self.en4(en3_out)                   #63, 256*256
        en5_out = self.en5(en4_out)                   #32, 512*512

        #################################
        #------------top down-----------#
        #################################  
        stage5 = self.lc5(en5_out)                         #32, 256 * 256
        stage4 = self.lc4(en4_out, stage5)                 #63, 256 * 256
        stage3 = self.lc3(en3_out, stage4)                 #125, 256 * 256
        stage2 = self.lc2(en2_out, stage3)                 #250, 256 * 256

        #################################
        #-----------aggregation---------#
        #################################  
        stage5 = self.f5(stage5)
        stage4 = self.f4(stage4)
        stage3 = self.f3(stage3)
        stage2 = self.f2(stage2)

        #################################
        #------------recovery-----------#
        #################################    
        stage5 = self.up5(stage5)
        if self.is_1000:
            stage5 = stage5[:,:,3:-3, 3:-3]
        stage4 = self.up4(stage4)
        if self.is_1000:
            stage4 = stage4[:,:,1:-1,1:-1]
        stage3 = self.up3(stage3)
        
        out = torch.cat([stage5, stage4, stage3, stage2], dim=1)
        out = self.recovery(out)

        return out

    def _make_layer(self, num, inc, outc, is_downsample=True, dilation=1):
        layers = []

        if is_downsample == True:
        #downsample and increase channel
            my_skip = nn.Sequential(
                nn.Conv2d(inc, outc, 1, stride=2, bias=False),
                nn.BatchNorm2d(outc)
            )
            layers.append(basicBlock(inc, outc, 2, skip=my_skip))
        else:
        #increase channel
            my_skip = nn.Sequential(
                nn.Conv2d(inc, outc, 1, stride=1, bias=False),
                nn.BatchNorm2d(outc)
            )
            layers.append(basicBlock(inc, outc, 1, skip=my_skip))

        for _ in range(num-1):
            layers.append(basicBlock(outc, outc, 1, dilation=dilation))

        return nn.Sequential(*layers)

class fpn50(nn.Module):
    def __init__(self, is_1000, inc=64, nums=[3, 4, 6, 3]):
        super(fpn50, self).__init__()
        self.en1 = nn.Sequential(
            nn.Conv2d(3, inc, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(inc),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.en2 = self._make_layer(nums[0], inc, inc, False)
        self.en3 = self._make_layer(nums[1], inc*4, inc*2)
        self.en4 = self._make_layer(nums[2], inc*8, inc*4)
        self.en5 = self._make_layer(nums[3], inc*16, inc*8)

        self.lc5 = convBnRelu(inc*32, 256, 1)
        
        if is_1000 == True:
            self.lc4 = lc(inc*16, True, True)
            self.lc3 = lc(inc*8, True, True)
        else:
            self.lc4 = lc(inc*16, True, False)
            self.lc3 = lc(inc*8, True, False)
        
        self.lc2 = lc(inc*4, True, False)

        self.f5 = nn.Sequential(
            convBnRelu(256, 128, 1, 1),
            convBnRelu(128, 128, 3, 1, 1)
        )
        self.f4 = nn.Sequential(
            convBnRelu(256, 128, 1, 1),
            convBnRelu(128, 128, 3, 1, 1)
        )
        self.f3 = nn.Sequential(
            convBnRelu(256, 128, 1, 1),
            convBnRelu(128, 128, 3, 1, 1)
        )
        self.f2 = nn.Sequential(
            convBnRelu(256, 128, 1, 1),
            convBnRelu(128, 128, 3, 1, 1)
        )

        self.up5 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.recovery = nn.Sequential(
            convBnRelu(512, 128, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            convBnRelu(128, 64, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 1, 3, 1, padding=1),
            nn.Sigmoid()
        )

        self.is_1000 = is_1000

    def forward(self, x):
        #################################
        #------------bottom up----------#
        #################################            
        en1_out = self.en1(x)                         #500, 64*64

        en2_out = self.en2(self.maxpool(en1_out))     #250, 64*64    
        en3_out = self.en3(en2_out)                   #125, 128*128
        en4_out = self.en4(en3_out)                   #63, 256*256
        en5_out = self.en5(en4_out)                   #32, 512*512

        #################################
        #------------top down-----------#
        #################################  
        stage5 = self.lc5(en5_out)                         #32, 256 * 256
        stage4 = self.lc4(en4_out, stage5)                 #63, 256 * 256
        stage3 = self.lc3(en3_out, stage4)                 #125, 256 * 256
        stage2 = self.lc2(en2_out, stage3)                 #250, 256 * 256

        #################################
        #-----------aggregation---------#
        #################################  
        stage5 = self.f5(stage5)
        stage4 = self.f4(stage4)
        stage3 = self.f3(stage3)
        stage2 = self.f2(stage2)

        #################################
        #------------recovery-----------#
        #################################    
        stage5 = self.up5(stage5)
        if self.is_1000:
            stage5 = stage5[:,:,3:-3, 3:-3]
        stage4 = self.up4(stage4)
        if self.is_1000:
            stage4 = stage4[:,:,1:-1,1:-1]
        stage3 = self.up3(stage3)
        
        out = torch.cat([stage5, stage4, stage3, stage2], dim=1)
        out = self.recovery(out)

        return out

    def _make_layer(self, num, inc, outc, is_downsample=True, dilation=1):
        layers = []

        if is_downsample == True:
        #downsample and increase channel
            my_skip = nn.Sequential(
                nn.Conv2d(inc, outc*4, 1, stride=2, bias=False),
                nn.BatchNorm2d(outc*4)
            )
            layers.append(bottleNeck(inc, outc, 2, skip=my_skip))
        else:
        #increase channel
            my_skip = nn.Sequential(
                nn.Conv2d(inc, outc*4, 1, stride=1, bias=False),
                nn.BatchNorm2d(outc*4)
            )
            layers.append(bottleNeck(inc, outc, 1, skip=my_skip))

        for _ in range(num-1):
            layers.append(bottleNeck(outc*4, outc, 1, dilation=dilation))

        return nn.Sequential(*layers)

class sobelFilter(nn.Module):
    '''
    apply sobel filter on input tensor
    output tensor with 2 channels
    '''

    def __init__(self, device):
        super(sobelFilter, self).__init__()

        self.sobel_x = torch.tensor([[1., 0., -1.], \
                            [2., 0., -2.], \
                            [1., 0., -1.]]).unsqueeze(0).unsqueeze(0).to(device)

        self.sobel_y = torch.tensor([[1.,   2.,  1.],\
                            [0.,   0.,  0.],\
                            [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, x):
        grad_out = F.pad(x, (1, 1, 1, 1), 'reflect')
        grad_out_x = F.conv2d(grad_out, self.sobel_x)
        grad_out_y = F.conv2d(grad_out, self.sobel_y)
        grad_out = torch.cat((grad_out_x, grad_out_y), 1)

        return grad_out

class PseudoEdgeNet(nn.Module):
    def __init__(self, device, is_1000):
        super(PseudoEdgeNet, self).__init__()
        self.segNet = fpn50(is_1000=is_1000)
        self.edgeNet = edgeNet()
        self.attention = fpn18(is_1000=is_1000)
        self.sobel = sobelFilter(device)

    def forward(self, x):

        seg_out = self.segNet(x)
        edge_out = self.edgeNet(x)
        attention_out = self.attention(x)

        edge_out = edge_out * attention_out
        grad_out = self.sobel(seg_out)

        return seg_out, edge_out, grad_out

def build_network(device, is_1000=False):
    return PseudoEdgeNet(device, is_1000=False)