#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adopted from https://github.com/Statistical-Downscaling-for-the-Ocean/graph-neural-net/blob/main/ by 
@author: rlc001
"""

import torch
import torch.nn.functional as F
import numpy as np
from losses import WeightedMSE
from model import FNO2d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
########### If you have torch_geometric ##############
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
# def make_snapshot_data(input_data, target_vals, mask):
#     T, C, S, D = input_data.shape
#     data_list = []
#     for t in range(T):
#         data = Data(
#             x=torch.tensor(input_data[t], dtype=torch.float32),
#             y=torch.tensor(target_vals[t], dtype=torch.float32),
#             mask=torch.tensor(mask[t], dtype=torch.float32)
#         )
#         data_list.append(data)
#     return data_list
###################################################
######## else maske custom torch dataset ##########

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class make_snapshot_data(Dataset):   ##Changed
    def __init__(self,input_data, target_vals, mask):
        self.input = input_data
        self.target = target_vals
        self.mask = mask
    def __getitem__(self, index):
        x=torch.tensor(self.input[index], dtype=torch.float32)
        y=torch.tensor(self.target[index], dtype=torch.float32)
        mask=torch.tensor(self.mask[index], dtype=torch.float32)
        return x, y, mask
    def __len__(self):
        return len(self.input)
    
#################################################

def evaluate_snapshot(model, loader, loss_function): ##Changed
    model.eval()
    batch_mse = 0
    with torch.no_grad():
        for x,ys,mask in loader:  ##NEW
            x = x.to(device)   ##NEW
            ys = ys.to(device)  ##NEW
            mask = mask.to(device)  ##NEW

            ys_pred = model(x)
            if mask.sum() == 0:
                continue
            batch_mse += loss_function(ys, ys_pred, mask)
    if batch_mse == 0:
        return np.nan
    mse = batch_mse / len(loader)
    return mse

def train_snapshot_model(model, train_loader, val_loader=None, lr=1e-3, wd=1e-5, epochs=200, reduction = 'mean_snap', save_path=None):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val_mse = np.inf
    best_state = None
    MSE_loss = WeightedMSE(reduction = reduction)  ##NEW

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x,y,mask  in train_loader:

            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            out = model(x)
            if mask.sum() == 0:
                continue
            loss = MSE_loss(out, y, mask, print_loss = True)  ##NEW
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # validation
        if val_loader is not None:
            val_mse = evaluate_snapshot(model, val_loader, MSE_loss)  ##Changed
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = model.state_dict()
                if save_path is not None:
                    torch.save(best_state, save_path)
                    print(f"Saved best model (val_mse={best_val_mse:.4f})")
                
        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep:03d} | TrainLoss={total_loss:.4f} | ValMSE={val_mse if val_loader else 'N/A'}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_model(data_train, data_val,width = 20, num_layers = 1, modes1 = None, modes2 = None, batch_size = 2, n_epochs=200, lr=1e-3, wd=1e-5, reduction = 'mean_snap',save_path=None):  ##Changed
    input_train, target_train, mask = data_train
    val_input_train, target_val, val_mask  = data_val

    T, C, S, D = input_train.shape  
    if modes1 is None: ##NEW
        modes1  = S  
    if modes2 is None:  ##NEW
        modes2 = np.floor(D/2) + 1 
    else:
        assert modes2 <= np.floor(D/2) + 1 

    train_dataset = make_snapshot_data(input_train, target_train, mask)  ##Changed
    val_dataset = make_snapshot_data(val_input_train, target_val, val_mask)  ##Changed

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model =  FNO2d(C, width , modes1, modes2, num_layers  =num_layers)  ##New
    model = train_snapshot_model(model, train_loader, val_loader, epochs=n_epochs, lr = lr, wd = wd,  reduction = reduction, save_path=save_path)   ##Changed
    
    return model