#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adopted from https://github.com/Statistical-Downscaling-for-the-Ocean/graph-neural-net/blob/main/ by
@author: rlc001
"""


import sys
import os
sys.path.append(os.path.dirname(__file__))

from data_processing import prepare_data
from train import train_model
from train import make_snapshot_data
from evaluate import evaluate_model
import torch
from pathlib import Path
########### If you have torch_geometric ##############
# from torch_geometric.loader import DataLoader
######## else maske manual torch dataset ##########
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
###############################################
from datetime import datetime
from argparse import ArgumentParser


def main(output_dir, data_dir):

    # === Prepare Data ===
    # main_dir = "/fs/site5/eccc/crd/ccrn/users/rpg002/stat_downscaling-workshop/FNO"
    # data_dir = "/fs/site5/eccc/crd/ccrn/users/rpg002/stat_downscaling-workshop/data" 
    
    main_dir = "/path/to/my/projects/line_p/"
    data_dir = "/path/to/my/projects/line_p/data/observation"
    
    data_dir = Path(data_dir)
    
    # === Prepare Data ===
    train_data, val_data, test_data, stations, depths = prepare_data(
        data_dir=data_dir,
        work_dir=work_dir,
        year_range=(1999, 2000),
        stations=["P22", "P23", "P24", "P25", "P26"],
        target_variable="Temperature",
        train_ratio=0.7, ##Changed
        val_ratio=0.15  ##Changed
    )

    # === Train Model ===
    try: ##Changed
        save_path = f"{work_dir}/best_model.pt"
        model = train_model(train_data, val_data, width = 20, num_layers = 1, modes1 = None, modes2 = None, batch_size = 2, n_epochs=200, lr=1e-3, wd=1e-5, reduction = 'mean_snap', save_path=save_path) ##Changed
    except Exception as e: ##Changed
        import shutil  ##Changed
        Path(output_dir + '/failed_cases').mkdir(parents=True, exist_ok=True)  ##Changed
        shutil.move(work_dir, output_dir + '/failed_cases')  ##Changed
        print("Terminated due to the follwoing error:\n", e)  ##Changed
        raise  #

    # === Evaluate Model ===
    model.load_state_dict(torch.load(save_path))
    input_test, target_test, mask_test = test_data
    test_data = make_snapshot_data(input_test, target_test, mask_test)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    _ = evaluate_model(
        model,
        test_loader,
        target_variable="Temperature",
        stations=stations,
        depths=depths,
        work_dir=work_dir
    )

if __name__ == "__main__":
    parser = ArgumentParser(description="Training and evaluation of downscaling Line P data using Fourier neural operator ")
    parser.add_argument("--output-directory", help="Directory to store artifacts from this script", default="/fs/site5/eccc/crd/ccrn/users/rpg002/stat_downscaling-workshop/FNO")
    parser.add_argument("--data-directory", help="Directory for the Line P training data", default="/fs/site5/eccc/crd/ccrn/users/rpg002/stat_downscaling-workshop/data")
    args = parser.parse_args()
    main(Path(args.output_directory), Path(args.data_directory))
