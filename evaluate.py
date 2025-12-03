#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adopted from https://github.com/Statistical-Downscaling-for-the-Ocean/graph-neural-net/blob/main/ by
@author: rlc001
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from losses import WeightedMSE
import xarray as xr
from pathlib import Path

def denormalize_variable(var_name, data_norm, scale_params):
    """Rescale normalized data back to original physical units."""
    params = scale_params[var_name]
    method = params["method"]

    if method == "zscore":
        return data_norm * params["std"] + params["mean"]
    elif method == "minmax":
        return data_norm * (params["max"] - params["min"]) + params["min"]
    else:
        return data_norm  # unchanged

def evaluate_model(model, test_loader, target_variable="Temperature", stations=None, depths=None, work_dir=Path(".")):

    plot_dir = work_dir / "plots"
    plot_dir.mkdir(Parents=True, exist_ok=True)
    units = {
        'Temperature': 'deg C',
        'Salinity': 'PSU',
        'Oxygen': 'umol/kg'
        }

    num_stations = len(stations)
    num_depths = len(depths)

    model.eval()

    ys_true_full = []
    ys_pred_full = []

    with torch.no_grad():
        for x, ys, mask in test_loader:  ##Changed
            x = x.to(model.device if hasattr(model, "device") else "cuda" if torch.cuda.is_available() else "cpu")  ##Changed
            ys = ys.to(model.device if hasattr(model, "device") else "cuda" if torch.cuda.is_available() else "cpu")  ##Changed
            mask = mask.to(model.device if hasattr(model, "device") else "cuda" if torch.cuda.is_available() else "cpu")  ##Changed

            out = model(x)  # [num_nodes]   ##Changed

            mask = torch.flatten(mask, start_dim = 1).cpu().numpy()        # [num_nodes]
            y_true = torch.flatten(ys, start_dim = 1).cpu().numpy()         # [num_nodes]
            y_pred = torch.flatten(out, start_dim = 1).cpu().numpy()            # [num_nodes]

            # Fill with NaN where mask is False
            y_true_filled = np.where(mask, y_true, np.nan)
            y_pred_filled = np.where(mask, y_pred, np.nan)

            ys_true_full.append(y_true_filled)
            ys_pred_full.append(y_pred_filled)

    ys_true_full = np.stack(ys_true_full, axis=0)  # [time, nodes]
    ys_pred_full = np.stack(ys_pred_full, axis=0)


    with open(work_dir / "scale_params_target.json") as f:
        scale_params_target = json.load(f)

    ys_pred_denorm = denormalize_variable(target_variable, ys_pred_full, scale_params_target)
    ys_true_denorm = denormalize_variable(target_variable, ys_true_full, scale_params_target)

    # Flatten only valid values for metrics
    valid_mask = ~np.isnan(ys_true_denorm)
    ys_true_valid = ys_true_denorm[valid_mask]
    ys_pred_valid = ys_pred_denorm[valid_mask]

    # --- Compute metrics only on valid points ---
    mse = mean_squared_error(ys_true_valid, ys_pred_valid)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(ys_true_valid, ys_pred_valid)
    r2 = r2_score(ys_true_valid, ys_pred_valid)
    bias = np.mean(ys_pred_valid - ys_true_valid)

    print("\n=== Model Evaluation Metrics ===")
    print(f"RMSE: {rmse:.4f} {units[target_variable]}")
    print(f"MAE : {mae:.4f} {units[target_variable]}")
    print(f"R²  : {r2:.4f}")
    print(f"Bias: {bias:.4f} {units[target_variable]}")
    print("================================\n")

    # 1. Scatter
    plt.figure(figsize=(5, 5))
    plt.scatter(ys_true_valid, ys_pred_valid, s=10, alpha=0.6)
    lims = [min(ys_true_valid.min(), ys_pred_valid.min()), max(ys_true_valid.max(), ys_pred_valid.max())]
    plt.plot(lims, lims, 'k--', lw=1)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"Scatter: RMSE={rmse:.3f} {units[target_variable]}, R²={r2:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "scatter_pred_vs_obs.png", dpi=150)

    # 2. Residual histogram
    plt.figure(figsize=(6, 4))
    plt.hist(ys_pred_valid - ys_true_valid, bins=40, color='gray', edgecolor='black', alpha=0.7)
    plt.title("Residual Distribution (Predicted - Observed)")
    plt.xlabel(f"Residual [{units[target_variable]}]")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plot_dir / "residual_hist.png", dpi=150)

    # 3. Depth–Station heatmap of mean residuals
    mean_resid = np.nanmean(ys_pred_full - ys_true_full, axis=0)  # mean across time
    resid_2d = mean_resid.reshape(num_stations, num_depths)
    resid_2d_plot = resid_2d.T # shape: (num_depths, num_stations)

    plt.figure(figsize=(6, 4))
    plt.imshow(resid_2d_plot, cmap='coolwarm', aspect='auto', origin='lower')
    plt.colorbar(label=f"Residual (Pred - Obs) [{units[target_variable]}]")
    if depths is not None:
        if type(depths) == xr.DataArray:
            depths = depths.values
        plt.yticks(range(num_depths), depths)
        plt.ylabel('depth')
    if stations is not None:
        if type(stations) == xr.DataArray:
            stations = stations.values
        plt.xticks(range(num_stations), stations, rotation=45)
        plt.xlabel('stations')
    plt.title("Mean Residual by Station/Depth")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plot_dir / "residual_heatmap.png", dpi=150)

    np.savez(work_dir / "output.npz", ys_true_full, ys_pred_full)

    return ys_true_full, ys_pred_full