import torch
import torch.nn as nn
import torch.nn.functional as F

class softmax(nn.Module):
    def __init__(self, smooth=1):
        super(softmax, self).__init__()
        self.smooth = smooth

    def forward(self, pred, true):
        bs = pred.shape[0]

        pred = torch.clamp(pred, 1e-7, 1-1e-7)
        pred = pred.view(bs, 2, -1)
        true = true.contiguous().view(bs, -1)
        one_dim_pred = pred[:, 1] > 0.5

        assert one_dim_pred.shape == true.shape        

        intersection = torch.sum(one_dim_pred * true, dim=1)
        union = torch.sum(one_dim_pred + true, dim=1) - intersection

        acc = (intersection+self.smooth)/(union+self.smooth)

        return torch.mean(acc)


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

        intersection = torch.sum(pred * true, dim=1)
        union = torch.sum(pred + true, dim=1) - intersection

        acc = (intersection+self.smooth)/(union+self.smooth)

        return torch.mean(acc)