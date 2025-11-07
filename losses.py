import numpy as np
import torch

import torch.nn.functional as F


class WeightedMSE:  ##NEW

    def __init__(self, weights = None, reduction='mean'):
        self.reduction = reduction
        if weights is None:
            weights = torch.tensor([1])
        self.weights = weights

    def __call__(self, data, target, mask = None, print_loss = False):


        y_hat = data
        y = target

        if mask is not None:
            weight =  mask * self.weights.to(data)
        else:
            weight = torch.ones_like(y) * self.weights.to(data)

        if self.reduction.lower() == 'mean_snap':
            mask_time = torch.tensor(weight.clone().sum((-1,-2)), dtype = bool)
            SE = ((y_hat - y)**2 * weight).sum((-1,-2))
            loss = ( SE[mask_time]/ weight.sum((-1,-2))[mask_time]).mean()
        elif self.reduction.lower() == 'mean':
            loss = (y_hat - y)**2 
            loss = (loss * weight).sum()/ weight.sum()
        elif self.reduction.lower() == 'sum':
            loss = (y_hat - y)**2 
            loss = torch.sum(loss * weight, dim = (-1,-2)).mean()

        if print_loss:
            print(f'MSE : {loss}')
        return loss
