import torch
import torch.nn as nn

class cons_loss(nn.Module):
    def __init__(self):
        super(cons_loss, self).__init__()

    def forward(self, pred):
        bs = pred.shape[0]
        ch = pred.shape[1]

        pred = torch.clamp(pred, 1e-7, 1-1e-7)
        pred = pred.view(bs, ch, -1)
     
        a = pred[:, :ch//2]
        b = pred[:, ch//2:]

        neg_loss = -(1-b) * torch.log(1-a)
        pos_loss = -b * torch.log(a)
        
        total_loss = neg_loss + pos_loss

        return 1-torch.mean(total_loss)

class cons_loss2(nn.Module):

    def __init__(self):
        super(cons_loss2, self).__init__()

    def forward(self, pred):
        bs = pred.shape[0]
        ch = pred.shape[1]

        pred = pred.view(bs, ch, -1)
     
        a = pred[:, :ch//2]
        b = pred[:, ch//2:]

        total_loss = torch.abs(a-b)
        total_loss = torch.min()

        return 1-torch.mean(total_loss)