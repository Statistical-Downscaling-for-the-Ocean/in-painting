#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adopted from https://github.com/Statistical-Downscaling-for-the-Ocean/graph-neural-net/blob/main by 
@author: rlc001
"""


import pandas as pd
import numpy as np
import glob
import os
import xarray as xr
import json

def load_ctd_data(data_dir, start_year, end_year):
    """
    Load and process CTD csv files for a given year range.
    Returns an xarray.Dataset with dimensions (depth, station, time)).
    """
    
    data_dir = f"{data_dir}/lineP_ctds"
    
    # Collect files by year
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    year_files = [
        f for f in all_files
        if any(str(y) in os.path.basename(f) for y in range(start_year, end_year + 1))
    ]

    if not year_files:
        raise FileNotFoundError(f"No csv files found for years {start_year}-{end_year} in {data_dir}")

    # Load and concatenate all data
    df_list = []
    for file in year_files:
        print(f"Loading {os.path.basename(file)} ...")
        df = pd.read_csv(file)
        df = df.rename(columns={
            "latitude": "Latitude",
            "longitude": "Longitude",
            "CTDTMP_ITS90_DEG_C": "Temperature",
            "SALINITY_PSS78": "Salinity",
            "OXYGEN_UMOL_KG": "Oxygen",
            "PRS_bin_cntr": "Depth",
        })
        df["time"] = pd.to_datetime(df["time"])
        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)

    # Sort and get unique coords
    depths = np.sort(df_all["Depth"].unique())
    stations = sorted(
        df_all["closest_linep_station_name"].unique(),
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
    times = np.sort(df_all["time"].unique())

    # Build arrays
    variables = ["Temperature", "Salinity", "Oxygen", "Latitude", "Longitude"]
    data_dict = {var: np.full((len(times), len(stations), len(depths)), np.nan) for var in variables}

    for t_idx, t in enumerate(times):
        df_t = df_all[df_all["time"] == t]
        for s_idx, s in enumerate(stations):
            df_s = df_t[df_t["closest_linep_station_name"] == s]
            if df_s.empty:
                continue
            depth_idx = np.searchsorted(depths, df_s["Depth"])
            for var in variables:
                valid = (depth_idx >= 0) & (depth_idx < len(depths))
                data_dict[var][t_idx, s_idx, depth_idx[valid]] = df_s[var].values[valid]
    
    # Return as xarray dataset
    ds = xr.Dataset(
        {
            var: (("time", "station", "depth"), data_dict[var]) for var in variables
        },
        coords={
            "time": times,
            "station": stations,
            "depth": depths
        },
    )
    ds["depth"].attrs["units"] = "m"
    ds["Temperature"].attrs["units"] = "deg C"
    ds["Salinity"].attrs["units"] = "PSS-78"
    ds["Oxygen"].attrs["units"] = "umol/kg"
    ds["Longitude"].attrs["units"] = "deg"
    ds["Latitude"].attrs["units"] = "deg"
        
    return ds



def normalize_dataset(ds, var_methods=None):
    """
    Normalize selected variables in an xarray.Dataset for ML.
    Returns:
      - normalized dataset
      - dictionary of scaling parameters for rescaling later
    """

    ds_norm = ds.copy(deep=True)
    scale_params = {}

    # Default normalization methods (can override with var_methods)
    default_methods = {
        "Temperature": "zscore",
        "Salinity": "minmax",
        "Oxygen": "zscore",
        "Bathymetry": "minmax",
        "Depth": "minmax",
        "Latitude": None,
        "Longitude": None,
    }

    if var_methods is None:
        var_methods = default_methods

    for var in ds.data_vars:
        method = var_methods.get(var, None)
        data = ds[var]

        if method == "zscore":
            mean_val = float(data.mean(skipna=True))
            std_val = float(data.std(skipna=True))
            ds_norm[var] = (data - mean_val) / std_val

            scale_params[var] = {
                "method": "zscore",
                "mean": mean_val,
                "std": std_val
            }

        elif method == "minmax":
            min_val = float(data.min(skipna=True))
            max_val = float(data.max(skipna=True))
            ds_norm[var] = (data - min_val) / (max_val - min_val)

            scale_params[var] = {
                "method": "minmax",
                "min": min_val,
                "max": max_val
            }

        else:
            # Variable not normalized (e.g., coordinates)
            scale_params[var] = {"method": None}
            continue

        print(f"Normalized {var} using {method}")

    return ds_norm, scale_params

def apply_normalization(ds, scale_params):
    """Apply precomputed normalization parameters to a dataset."""
    ds_norm = ds.copy(deep=True)
    for var, params in scale_params.items():
        if params["method"] == "zscore":
            mean_val = params["mean"]
            std_val = params["std"]
            ds_norm[var] = (ds[var] - mean_val) / std_val

        elif params["method"] == "minmax":
            min_val = params["min"]
            max_val = params["max"]
            ds_norm[var] = (ds[var] - min_val) / (max_val - min_val)
        # else: leave unchanged
    return ds_norm

def make_synthetic_linep(time, stations, depths) -> xr.Dataset:
   
    T = len(time)
    D = len(depths)
    S = len(stations)
    rng = np.random.default_rng(0)
    data = np.zeros((T, S, D), dtype=np.float32)

    for ti, t in enumerate(time):
        seasonal = 4.0 * np.sin(2 * np.pi * (t.dt.month - 1) / 12.0)
        for si in range(S):
            for di, depth in enumerate(depths):
                val = seasonal
                val += 0.2 * si                         
                val += np.exp(-depth / 200.0)          
                val += 0.3 * np.sin(0.1 * si * ti / max(1, S))
                val += 0.5 * rng.normal()             
                data[ti, si, di] = val + 10

    ds = xr.Dataset({"Temperature": (("time", "station", "depth"), data)}, coords={"time": time, "station": stations, "depth": depths})

    return ds

def reshape_to_tcsd(ds_input: xr.DataArray, ds_target: xr.DataArray):    ##NEW
    ds_input = xr.concat([ds_input[var] for var in list(ds_input.data_vars)], dim = 'channels')
    ds_target = xr.concat([ds_target[var] for var in list(ds_target.data_vars)], dim = 'channels')
    mask = (~np.isnan(ds_target)).astype(int)
    return (ds_input.fillna(0).to_numpy(), ds_target.fillna(0).to_numpy(), mask.to_numpy())


#%%

def prepare_data(
    work_dir: str,
    data_dir: str,   ##Changed
    year_range: tuple[int, int],
    stations: list[str] | None = None,
    # depths: list[float] | None = None,  ##Changed
    target_variable: str = "Temperature",
    bathymetry_in : xr.DataArray | None = None,  ##Changed
    train_ratio = 0.7,  ##Changed
    val_ratio = 0.15   ##Changed

):
    
    #work_dir = "/home/rlc001/data/ppp5/analysis/stat_downscaling-workshop"
    #year_range = (1999, 2000)
    #variable = "Temperature"
    #stations = ["P22", "P23", "P24", "P25", "P26"]
    #depths = [0.5, 10.5, 50.5, 100.5]
    
    start_year, end_year = year_range
    ds = load_ctd_data(data_dir, start_year, end_year)
    
    # Subset stations and depths
    #print(ds.station.values)
    if stations is not None: 
        ds = ds.sel(station=stations)

    #### For now to test but to be removed later ####
    depths = [0.5, 10.5, 50.5, 100.5]     ##Changed
    ds = ds.sel(depth=depths)   ##Changed
    #################################################

    
    ds_target = ds[[target_variable]]
    stations = ds_target['station']
    depths = ds_target['depth']
    ds_target = ds_target.expand_dims('channels', axis = -3)
    
    # Generate synthetic line p temperature 'model' data
    # Replace this by loading model data
    ds_input = make_synthetic_linep(ds_target['time'], ds_target['station'], ds_target['depth'])
    ds_input = ds_input.expand_dims('channels', axis = -3)
    # Add static variables
    if bathymetry_in is None:    ##Changed
        bathymetry_in = (~np.isnan(ds_input)).astype(int).rename({target_variable : 'Bathymetry'})   ##Changed

    # ds_input = ds_input.fillna(0)
    ds_input["Bathymetry"] = bathymetry_in["Bathymetry"]
    # ds_input = xr.concat([ds_input[target_variable], bathymetry_in['Bathymetry']], dim = 'channels')

    depth_in = xr.DataArray(
    ds_target.depth.values,
    dims=("depth",),
    coords={"depth": ds_input.depth},
    name="Depth"
    )
    ds_input["Depth"] = depth_in.broadcast_like(ds_input[target_variable])
    
    # === Split Data into train, validation, test ===
    T = ds_input.sizes["time"]
    # split ratios
    # split indices
    train_end = int(train_ratio * T)
    val_end = int((train_ratio + val_ratio) * T)
    
    ds_input_train = ds_input.isel(time=slice(0, train_end))
    ds_input_val   = ds_input.isel(time=slice(train_end, val_end))
    ds_input_test  = ds_input.isel(time=slice(val_end, T))
    
    ds_target_train = ds_target.isel(time=slice(0, train_end))
    ds_target_val   = ds_target.isel(time=slice(train_end, val_end))
    ds_target_test  = ds_target.isel(time=slice(val_end, T))

    # Normalization
    # Compute scale parameters from training data and apply to validation and test
    ds_input_train_norm, scale_params_in = normalize_dataset(ds_input_train)
    # Save input normalization parameters
    with open(f"{work_dir}/scale_params_in.json", "w") as f:
        json.dump(scale_params_in, f, indent=2)
    
    # Apply same normalization to validation & test inputs
    ds_input_val_norm  = apply_normalization(ds_input_val, scale_params_in)
    ds_input_test_norm = apply_normalization(ds_input_test, scale_params_in)
    
    ds_target_train_norm, scale_params_target = normalize_dataset(ds_target_train)
    # Save target normalization parameters
    with open(f"{work_dir}/scale_params_target.json", "w") as f:
        json.dump(scale_params_target, f, indent=2)
    
    # Apply same normalization to validation & test targets
    ds_target_val_norm  = apply_normalization(ds_target_val, scale_params_target)
    ds_target_test_norm = apply_normalization(ds_target_test, scale_params_target)

    # reshape data into graph structure, and compute target value mask
    print("\nPrepare Training:")
    train_data = reshape_to_tcsd(ds_input_train_norm, ds_target_train_norm)  ##Changed
    print("Done")
    print("\nPrepare Validation:")  
    val_data = reshape_to_tcsd(ds_input_val_norm, ds_target_val_norm)   ##Changed
    print("Done")
    print("\nPrepare Testing:")
    test_data = reshape_to_tcsd(ds_input_test_norm, ds_target_test_norm)   ##Changed
    print("Done")

    return train_data, val_data, test_data, stations, depths 