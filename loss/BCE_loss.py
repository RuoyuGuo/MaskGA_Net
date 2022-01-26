import torch
import torch.nn as nn
import torch.nn.functional as F

class softmax(nn.Module):
    def __init__(self):
        super(softmax, self).__init__()

    def forward(self, pred, true):
        bs = pred.shape[0]

        pred = torch.clamp(pred, 1e-7, 1-1e-7)
        pred = pred.view(bs, 2, -1)
        true = true.contiguous().view(bs, -1)

        #compute negative loss and postive loss, respectively
        neg_loss = -(1-true) * torch.log(pred[:, 0])
        pos_loss = -true * torch.log(pred[:, 1])

        total_loss = neg_loss + pos_loss

        return torch.mean(total_loss)

class full(nn.Module):
    def __init__(self):
        super(full, self).__init__()

    def forward(self, pred, true):
        bs = pred.shape[0]

        pred = torch.clamp(pred, 1e-7, 1-1e-7)
        pred = pred.view(bs, -1)
        true = true.contiguous().view(bs, -1)

        #compute BCE loss
        total_loss = - (1-true) * torch.log(1-pred) - true * torch.log(pred)

        return torch.mean(total_loss)


class partial(nn.Module):
    '''
    neg: voronoi boundary
    pos: point annotations 
    '''
    def __init__(self, w1=1, w2=0.1):
        super(partial, self).__init__()
        self.w1 = w2
        self.w2 = w2

    def forward(self, pred, true):
        bs = pred.shape[0]

        pred = torch.clamp(pred, 1e-7, 1-1e-7)
        pred = pred.view(bs, -1)
        true = true.contiguous().view(bs, -1)
        pos_true = (true == 1).float()
        neg_true = (true == 2).float()

        #compute negative loss and postive loss, respectively
        pos_true_count = torch.sum(pos_true, 1)
        neg_true_count = torch.sum(neg_true, 1)

        pos_loss = torch.sum(-pos_true*torch.log(pred), 1)/pos_true_count
        neg_loss = torch.sum(-neg_true*torch.log(1-pred), 1)/neg_true_count

        total_loss = self.w1*pos_loss + self.w2*neg_loss

        return torch.mean(total_loss)