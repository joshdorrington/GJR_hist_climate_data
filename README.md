# GJR_hist_climate_data
This is data and code used in the paper 'How well does CMIP6 capture the dynamics of Euro-Atlantic weather regimes, and why?' by Josh Dorrington, Kristian Strommen and Federico Fabiano.


This repo contains:

### Code

* code/regime_computations_example.ipynb is a Jupyter notebook demonstrating how to compute classical and geopotential-jet regimes, how to easily cluster subsets of a time series, and how to compute the stability metric.
'ClusterBuster' is a set of Python code with an accompanying example notebook showing how to compute geopotential jet regimes in ERA20C, and compute regime stability.

* An example of how to perform the ridge regression can be found at https://github.com/fedef17/FScripts/blob/master/kj_multireg_hist_3n_ensmean.py

### Data

* regime_data/ contains regime patterns (as Z500 anomalies) and state sequences for both classical circulation regimes (CCR) and geopotential-jet regimes (GJR) for cluster numbers between K=2 and K=10. The focus of the paper is on GJR_K3, but all data is included for completeness.
* tuttecose_wcmip5.pkl - A python pickle file containing dictionaries of climate model predictors and regime metrics, used to fit the ridge regression model, and produce the scatter plots. 


## For clarification or technical help, or if you spot an error in the data/code, please contact joshua.dorrington@kit.edu

