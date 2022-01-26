import torch
import torch.nn as nn
import torch.nn.functional as F

class partial(nn.Module):

    def __init__(self, w1=1, w2=0.1):
        super(partial, self).__init__()
        self.w1 = w1
        self.w2 = w2

    def forward(self, pred, true):
        bs = pred.shape[0]

        pred = pred.view(bs, -1)
        true = true.contiguous().view(bs, -1)
        pos_true = (true == 1).float()
        neg_true = (true == 2).float()

        #compute the number of negative and postive pixels
        pos_true_count = torch.sum(pos_true, 1)
        neg_true_count = torch.sum(neg_true, 1)

        pos_loss = torch.sum(torch.abs(pos_true-pred*pos_true), dim=1)/pos_true_count
        neg_loss = torch.sum(torch.abs(neg_true-pred*neg_true), dim=1)/neg_true_count

        total_loss = self.w1*pos_loss + self.w2*neg_loss

        return torch.mean(total_loss)

class vanilla(nn.Module):

    def __init__(self):
        super(vanilla, self).__init__()

    def forward(self, pred, true):
        
        total_loss = torch.abs(pred - true)

        return torch.mean(total_loss)



