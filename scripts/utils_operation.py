import numpy as np
import torch

def regulate(x, eps_min=0.0, eps_max=1.0):
    x = -torch.nn.functional.relu(-x+eps_max)+eps_max 
    x = torch.nn.functional.relu(x-eps_min)+eps_min
    return x

def cumsum(x, triu_mask, dim=-2):
    N = x.size(-2)
    triu_mask = triu_mask[:N, :N].to(x.device)
    cumsum_x = (triu_mask[None, None, None, :, :] * x).sum(dim=dim, keepdim=True) # [1, TH, TW, 1, B]
    cumsum_x = cumsum_x.transpose(-1,-2) # [1, TH, TW, B, 1]

    return cumsum_x

def cumprod(x, triu_mask, dim=-2, eps_min=1e-8):
    x = torch.nn.functional.relu(x-eps_min)+eps_min 
    return torch.exp(cumsum(torch.log(x), triu_mask, dim=dim))



if __name__ == '__main__':
    pass