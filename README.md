# FNO
This repository contains the code related to an FNO approach to filling in Line-P data.



Work flow for a FNO from main.py.

The general idea is to bias correct the model output to observations where they are available.

## Notes
The code currently uses a single variable (temperature) and static variables (ex. bathymetry, depth ...) to predict a single target variable (observe temperature from ctds).
You can select a subset of the stations for testing.

## First-Time setup instructions
First clone down the repository
```
git clone https://github.com/Statistical-Downscaling-for-the-Ocean/fourier-neural-operator.git
```

Assuming a fresh python environment (via venv, conda, or pyenv), install all the needed packages
```
cd fourier-neural-operator
pip install -r requirements.txt
```

Make a data directory where you will store all the training data
```
mkdir /path/to/data/directory # example ~/data/linep
```

Download the Line P CTD data used for training:
```
cd /path/to/data/directory
wget https://hpfx.collab.science.gc.ca/dfo/SD-Ocean/Training/observations/ctd/lineP_CTD_training.csv
```

## Data processing
In main.py the "prepare_data" function
- Loads the target observations (Line P ctd observations, files LineP_ctds_YYYY_binned_1m.csv, function load_ctd_data).
- Loads the model data predictors (For now synthetic lineP data is generated in place of real model data).
- Splits the the data into training, validation and testings sets.
- Normalizes all sets of data with scaling parameters computed from only the training set (scale_params.json files are saved with the scaling parameters to denormalize later).
- Reshapes the data to appropriate batch x channel x stations x depth structure (reshape_to_tcsd).

To run this script:
```
python main.py --output-directory <path_to_output> --data-directory <path_to_data_directory>
```
Loads the target observations (Line P ctd observations, function load_ctd_data, in the proto type only 4 depth points are loaded).
Loads the model data predictors (For now synthetic lineP data is generated in place of real model data).
Splits the the data into training, validation and testings sets.
Normalizes all sets of data with scaling parameters computed from only the training set (scale_params.json files are saved with the scaling parameters to denormalize later).
Reshapes the data to appropriate batch x channel x stations x depth structure (reshape_to_tcsd).

## Model details
For this simple case there is no time dependency in the model. The model is defined in model.py as FNO2d.

You should choose model hyperparameters for:

1. **modes1** and **modes2** — the truncation in the spectral domain for the two dimensions of the data, which in this setup has a maximum of the number of stations and ⌊N/2⌋ + 1, with *N* being the number of depth points.
2. **Width** of the FNO blocks (number of channels)
3. **Number** of FNO blocks to stack on top of each other

## Architecture details: 
The FNO2d used Fourier Neural Operator blocks (Li et al, 2020: https://arxiv.org/abs/2010.08895). Each block transforms the input to the spectral domain using FFT, truncates at some spectral frequency, performs channel-wise transformation, inverses FFT the ourput, sums to the a linear transformation of the input, and passes the final tensor to an activation function. This architecture effectively learns dependence across spatial scales uisng an operator which is in-sensitive to sampling resolution. 

## Training

The model training is done in train.py with MSE as training criterion. The loss is only computed where there are valid observations. You can choose the reduction parameter to specify how the MSE is calucalted. The difault calculates MSE for each snapshot and then averages across samples.

## Evaluation
evalute_model in evaluate.py generates predictions from the testing data and compares them to valid observations and creates some plots.