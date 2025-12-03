#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adopted from https://github.com/Statistical-Downscaling-for-the-Ocean/graph-neural-net/blob/main by
@author: rlc001
"""


import pandas as pd
import numpy as np
import xarray as xr

import json

from pathlib import Path

def load_ctd_data(ctd_data_file, start_year, end_year):
    """
    Load and process CTD csv files for a given year range.
    Returns an xarray.Dataset with dimensions (depth, station, time)).
    """
    
    df_all = pd.read_csv(data_dir, comment="#")

    df_all["TIME"] = pd.to_datetime(df_all["TIME"], format="%Y-%m-%d %H:%M:%S")
    df_all = df_all.rename(
        columns={
            "LATITUDE": "Latitude",
            "LONGITUDE": "Longitude",
            "TEMPERATURE": "Temperature",
            "SALINITY": "Salinity",
            "OXYGEN_UMOL_KG": "Oxygen",
            "PRESSURE_BIN_CNTR": "Depth",
            "TIME": "time",
            "STATION_ID": "station",
        }
    )

    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year+1}-01-01")
    df_all = df_all[(df_all["time"] >= start_date) & (df_all["time"] < end_date)]

    # Sort and get unique coords
    depths = np.sort(df_all["Depth"].unique())
    stations = sorted(
        df_all["station"].unique(),
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
    stations.remove('P26')
    stations.append('P26')

    times = np.sort(df_all["time"].unique())
    longitudes = np.array([df_all[df_all['station'] == s]['Longitude'].mean() for s in stations])
    latitudes = np.array([df_all[df_all['station'] == s]['Latitude'].mean() for s in stations])
    distances = np.array([0 if ind == 0 else haversine(latitudes[ind], longitudes[ind], latitudes[ind - 1], longitudes[ind - 1]) for ind, s in enumerate(stations)])
     # # Build arrays
    variables = ["Temperature", "Salinity", "Oxygen", "Latitude", "Longitude"]
    data_dict = {var: np.full((len(times), len(stations), len(depths)), np.nan) for var in variables}

    for t_idx, t in enumerate(times):
        df_t = df_all[df_all["time"] == t]
        for s_idx, s in enumerate(stations):
            df_s = df_t[df_t["station"] == s]
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
            "lat" : latitudes,
            "lon" : longitudes,
            "distance" : distances,
            "depth": depths
        },
    )

    print(ds)

    ds["depth"].attrs["units"] = "m"
    ds["Temperature"].attrs["units"] = "deg C"
    ds["Salinity"].attrs["units"] = "PSU"
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
    work_dir: Path,
    data_dir: Path,   ##Changed
    year_range: tuple[int, int],
    stations: list[str] | None = None,
    depths = [0.5, 10.5, 50.5, 100.5],     ##Changed
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

    ctd_filename = data_dir / "lineP_CTD_training.csv"
    start_year, end_year = year_range
    ds = load_ctd_data(ctd_filename, start_year, end_year)

    # Subset stations and depths
    #print(ds.station.values)
    if stations is not None:
        station_numbers = [s.lstrip("P") for s in stations]
        print(station_numbers)
        print(ds.station)
        ds = ds.sel(station=station_numbers)

    #### For now to test but to be removed later ####
    print('==========================================================\n'+
        'Warning! In this protocode only 4 depth points are selcted! Edit for the actual training! \n' + 
        '==========================================================\n')
    depths = [0.5, 25.5, 50.5, 75.5]     ##Changed
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

    ds_target_train = ds_target.isel(time=slice(0, train_end))
    ds_target_val   = ds_target.isel(time=slice(train_end, val_end))

    if train_ratio + val_ratio < 1:
        ds_input_test  = ds_input.isel(time=slice(val_end, T))
        ds_target_test  = ds_target.isel(time=slice(val_end, T))
    else:
        print('==========================================================\n'+
              'Test split ratio is zero. Test set is the same as validation set! \n' + 
              '==========================================================\n')
        ds_input_test  = ds_input_val.copy()
        ds_target_test  = ds_target_val.copy()

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




def haversine(la0,lo0,la1,lo1):
    """ haversine formula with numpy array handling
    Calculates spherical distance between points on Earth in meters
    Compares elements of (la0,lo0) with (la1,lo1)
    Shapes must be compatible with numpy array broadcasting
    args: lats and lons in decimal degrees
    returns: distance on sphere with volumetric mean Earth radius in meters
    """
    rEarth=6371*1e3 # 
    # convert to radians
    la0=np.radians(la0)
    la1=np.radians(la1)
    lo0=np.radians(lo0)
    lo1=np.radians(lo1)
    theta=2*np.arcsin(np.sqrt(np.sin((la0-la1)/2)**2+np.cos(la0)*np.cos(la1)*np.sin((lo0-lo1)/2)**2))
    d=rEarth*theta
    return d
