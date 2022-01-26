import torch
from torch import nn
from torch.nn import functional as F

class sigmoid(nn.Module):
    def __init__(self, smooth=1):
        super(sigmoid, self).__init__()
        self.smooth = smooth

    def forward(self, pred, true):
        bs = pred.shape[0]

        pred = torch.clamp(pred, 1e-7, 1-1e-7)
        pred = pred.view(bs, -1)
        pred = pred > 0.5
        
        true = true.contiguous().view(bs, -1)   

        intersection = (pred * true).sum()                            
        dice = (2.*intersection + self.smooth)/(pred.sum() + true.sum() + self.smooth)  

        return torch.mean(dice)