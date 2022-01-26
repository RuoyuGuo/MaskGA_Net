import torch
import torch.nn as nn
import numpy as np 
import scipy.stats as st

from torch.nn import functional as F

class RSV(nn.Module):
    def __init__(self, device, hsize=64, sigma=40, iters=5):
        super(RSV, self).__init__()
        self.kernel =  self._gkern(hsize, sigma).to(device)
        self.eps = 1e-5
        self.iters = iters
        
    def forward(self, pred, mask):
        for i in range(self.iters):
            M_w_pre = 0 if i == 0 else M_w
            M_bar = 1 - mask + M_w_pre
    
            M_w = F.conv2d(M_bar, self.kernel, stride=1, padding='same')
            M_w = M_w * mask

        M_w = M_w_pre / (M_w+self.eps)
        
        total_loss = torch.abs(pred - mask)
        
        return torch.mean(total_loss)
    
    def _gkern(self, kernlen, nsig):
        """Returns a 2D Gaussian kernel."""

        x = np.linspace(-nsig, nsig, kernlen+1, dtype=np.float32)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        
        return torch.tensor(kern2d/kern2d.sum(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
