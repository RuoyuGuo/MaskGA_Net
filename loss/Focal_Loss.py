'''
Implementation of Focal loss and Noise Suppression Focal Loss
Focal Loss: Focal Loss for Dense Object Detection
Noise Suppression Focal Loss: PGD-UNet: A Position-Guided Deformable Network for Simultaneous Segmentation of Organs and Tumors

'''
import torch

from torch import nn
from torch.nn import functional as F

class FL(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2):
        '''
        focal loss for multi-class classfication, set alpha to [1, 1, ...], gamma to 0, 
        equals to cross entropy loss

        Parameter
        ---------
        @alpha: class balance weight, for binary classification, alpha is the weight
                for class 0, 1-alpha is the weight for class 1
                default value for rare class (class 0)
        @gamma: focusing parameter, greater value focuses more on hard exmaples
                exponent for factor
        @num_classes: number of classes 
        '''
        super(FL, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
            assert len(alpha) == num_classes, 'For multiclass classifcation, there should be {num_classes} elements in alpha'
        else:
            self.alpha = torch.tensor([alpha, 1-alpha])

    def forward(self, y_pred, y_true):
        '''
        y_pred: B * C * H * W
        y_true: B * C * H * W
        '''
        bs = y_pred.shape[0]
        self.alpha = self.alpha.to(y_pred.device)

        y_pred = torch.clamp(y_pred, 1e-7, 1-1e-7)

        if self.num_classes == 2:
            y_pred = y_pred.view(bs, -1).unsqueeze(1)
            y_true = y_true.contiguous().view(bs, -1).unsqueeze(1)
            y_pred = torch.cat((y_pred, 1-y_pred), dim=1)
            y_true = torch.cat((y_true, 1-y_true), dim=1)
        else:
            y_pred = y_pred.view(bs, self.num_classes, -1)
            y_true = y_true.contiguous().view(bs, self.num_classes, -1)
        
        #cross entropy
        ce_loss = -y_true * torch.log(y_pred)
        #factor
        factor = self.alpha[None,:,None]*torch.pow((1-y_pred), self.gamma)

        #total_loss
        total_loss = factor * ce_loss 
        total_loss = torch.sum(total_loss, dim=1)

        return torch.mean(total_loss)

class NSFL(nn.Module):
    def __init__(self, alpha=0.25, beta=0.2, gamma=2, epsilon=0.2, num_classes=2):
        '''
        Noise Suppression Focal Loss for multi-class classfication, set alpha to [1, 1, ...], gamma to 0, 
        equals to cross entropy loss

        Parameter
        ---------
        @alpha: class balance weight, for binary classification, alpha is the weight
                for class 0, 1-alpha is the weight for class 1
                default value for rare class (class 0)
        @beta: exponent of factor when pred < epsilon
        @gamma: focusing parameter, greater value focuses more on hard exmaples
                exponent for factor when pred > epsilon
        @epsilon: threshold to decide to use NSFL or FL 
        @num_classes: number of classes 
        '''
        super(NSFL, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_classes = num_classes
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
            assert len(alpha) == num_classes, 'For multiclass classifcation, there should be {num_classes} elements in alpha list'
        else:
            self.alpha = torch.tensor([alpha, 1-alpha])

    def forward(self, y_pred, y_true):
        '''
        y_pred: B * C * H * W
        y_true: B * C * H * W
        '''
        bs = y_pred.shape[0]
        self.alpha = self.alpha.to(y_pred.device)

        y_pred = torch.clamp(y_pred, 1e-7, 1-1e-7)

        if self.num_classes == 2:
            y_pred = y_pred.view(bs, -1).unsqueeze(1)
            y_true = y_true.contiguous().view(bs, -1).unsqueeze(1)
            y_pred = torch.cat((y_pred, 1-y_pred), dim=1)
            y_true = torch.cat((y_true, 1-y_true), dim=1)
        else:
            y_pred = y_pred.view(bs, self.num_classes, -1)
            y_true = y_true.contiguous().view(bs, self.num_classes, -1)
        
        #cross entropy
        ce_loss = -y_true * torch.log(y_pred)

        #threshold
        threshold = y_pred >= self.epsilon

        #factor
        factor_fl = self.alpha[None,:,None] * torch.pow((1-y_pred), self.gamma) * threshold
        factor_nsfl = ((1-self.epsilon)**self.gamma)/(self.epsilon**self.beta) * \
                                torch.pow((y_pred), self.beta) * \
                                ~threshold
 
        #total_loss
        total_loss = factor_fl * ce_loss + factor_nsfl * ce_loss
        total_loss = torch.sum(total_loss, dim=1)

        return torch.mean(total_loss)