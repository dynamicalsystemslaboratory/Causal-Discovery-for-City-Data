# Causal-discovery-for-city-data

## Overview
The folders and files are:

1. **FsATE_urban**: a folder includes the code (CMI_estimator.py file) to estimate the transfer entropy using binning method and infer the causal dependencies using permutation tests for transfer entropy or conditional transfer entropy.
- function _FsATE_parallel_shuffles_significance_ outputs the significance test of $\text{FsATE}_{X\rightarrow Y}$ by shuffling only $X_t$. 
- function _FsACTE_parallel_shuffles_significance_ outputs the significance test of $\text{FsACTE}_{X\rightarrow Y|Z}$ by shuffling $X_t$ while preserving the associations between $Y_t$ and $Z_t$. 
2. **test_synthetic.py**: a .py file that can generate synthetic data following a bivariate linear system and infer the causal associations.
3. **test_carbon.python**: a .py file that can be used to conduct causal analyses on _climate change_ data through features-adjusted transfer entropy (FsATE). 
4. **test_covid.python**: a .py file that can be used to conduct causal analyses on _infectious disease_ data through features-adjusted conditional transfer entropy (FsACTE). 
5. **carbon_processing.ipynb**: a .ipynb file that can be executed with _climate change_ dataset to visualize the scaling results. It can also be used to show the results of statistical tests (Kolmogorov-Smirnov test and Moran's I) for whether it is independently and identically distributed.
5. **covid_processing.ipynb**: a .ipynb file that can be executed with _infectious disease_ dataset to visualize the scaling results. It can also be used to show the results of statistical tests (Kolmogorov-Smirnov test and Moran's I) for whether it is independently and identically distributed.
6. **Data**: a folder includes the real-world datasets.

## Required python packages 
- python
- numpy
- numba
- scipy
- statsmodels
- pandas
- geopandas
