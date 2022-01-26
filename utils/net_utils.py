import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3x3(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class Conv3x3Drop(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3Drop, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.Dropout(p=0.2),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class Conv3x3Small(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3Small, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.ELU(),
                                   nn.Dropout(p=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.ELU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # self.deconv = nn.ConvTranspose2d(in_feat, out_feat,
        #                                  kernel_size=3,
        #                                  stride=1,
        #                                  dilation=1)

        self.deconv = nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        # outputs = self.up(inputs)
        outputs = self.deconv(inputs)
        out = torch.cat([down_outputs, outputs], 1)
        return out


class UpSample(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpSample, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.deconv = nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        outputs = self.up(inputs)
        # outputs = self.deconv(inputs)
        out = torch.cat([outputs, down_outputs], 1)
        return out

class convBnRelu(nn.Module):
    '''
    conv2d -> batch normalizatoin -> relu
    
    Parameters
    ----------
    @inc: input channel
    @outc: output channel
    @k: kernel_size
    @s: stride
    @p: padding
    '''

    def __init__(self, inc, outc, k, s=1, p=0):
        super(convBnRelu, self).__init__()
        self.module = nn.Sequential(
                    nn.Conv2d(inc, outc, k, s, padding=p, bias=False),
                    nn.BatchNorm2d(outc),
                    nn.ReLU())
    
    def forward(self, x):

        return self.module(x)

class convGnRelu(nn.Module):
    '''
    conv2d -> Group normalizatoin -> relu
    
    Parameters
    ----------
    @inc: input channel
    @outc: output channel
    @k: kernel_size
    @s: stride
    @p: padding
    '''

    def __init__(self, inc, outc, k, s=1, p=0):
        super(convGnRelu, self).__init__()
        self.module = nn.Sequential(
                    nn.Conv2d(inc, outc, k, s, padding=p, bias=False),
                    nn.GroupNorm(32, outc),
                    nn.ReLU())
    
    def forward(self, x):

        return self.module(x)

class basicBlock(nn.Module):
    '''
    build basic block in residual net, 2 layers
    
    Parameters
    ----------
    @inc: input channel
    @outc: output channel
    @stride: stride
    @dilation: dilation value
    @skip: skip connection method
    '''
    
    def __init__(self, inc, outc, stride=1, dilation=1, skip=None):
        super(basicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inc, outc, 3, stride=stride, padding=dilation, \
                                dilation= dilation, bias=False)
        self.nm1 = nn.GroupNorm(32, outc)
        self.conv2 = nn.Conv2d(outc, outc, 3, padding=1, bias=False)
        self.nm2 = nn.GroupNorm(32, outc)
        self.relu = nn.ReLU()

        self.skip = skip

    def forward(self, x):
        shortcut = x

        out = self.relu(self.nm1(self.conv1(x)))
        out = self.nm2(self.conv2(out))

        if self.skip is not None:
            shortcut = self.skip(x)

        out += shortcut
        out = self.relu(out)

        return out
        
class bottleNeck(nn.Module):
    '''
    build bottle_neck in residual net, 3 layers
    
    Parameters
    ----------
    @inc: input channel
    @outc: output channel
    @stride: stride
    @dilation: dilation value
    @skip: skip connection method
    '''

    def __init__(self, inc, outc, stride=1, dilation=1, skip=None):
        super(bottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inc, outc, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, stride=stride, padding=dilation, \
                                            dilation=dilation, bias=False)  
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