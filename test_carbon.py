
from FsATE_urban.CMI_estimator import CMI_estimator
import numpy as np
from scipy.stats import pearsonr, spearmanr
# from itertools import combinations
from scipy.integrate import odeint
import scipy.io
from scipy import stats, linalg

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.signal import detrend


def adf_detrend(data):
    # ADF Test on each column
    N, T = data.shape
    ts = np.zeros((N,T),dtype=float)
    for n in range(N):
        result = adfuller(data[n])
        if result[1]>0.05:
            ts_detrended = detrend(data[n])
        # result_detrended = adfuller(ts_detrended)
        
        else: ts_detrended = data[n]
        ts[n] = ts_detrended.reshape(1,T)
    return ts


################### CO2 emission ################### 

FsAMIs_CO2 = np.load('./Scaling_Results/FAMIs_CO2.npy')
FsAMIs_TEMP = np.load('./Scaling_Results/FAMIs_TEMP.npy')

N,T = FsAMIs_TEMP.shape
print(N,T)

FsAMIs_CO2_dt = adf_detrend(FsAMIs_CO2)
FsAMIs_TEMP_dt = adf_detrend(FsAMIs_TEMP)

all = [FsAMIs_CO2_dt,FsAMIs_TEMP_dt]
array_FsAMI = np.zeros((N,len(all),T),dtype=float)

array_FsAMI[:,0,:] = FsAMIs_CO2_dt
array_FsAMI[:,1,:] = FsAMIs_TEMP_dt

var_names = ['$CO2$', '$TEMP$']


#########################
significance = 0.05
tau_max = 4 # time history
tau_lag = 1 # time lag
sig_samples = 10000; symbs = 4

array = array_FsAMI
print('array_FAMI', var_names, array.shape)

cmi = CMI_estimator(array=array, sig_samples=sig_samples, symbs = symbs)
for i in range(array.shape[1]):
    for j in range(array.shape[1]):
        if i!=j:    
            Y = [(j, 0)]
            Z  = [(j, -t) for t in range(1, tau_max+1)]
            X = [(i, -tau_lag)]; 
            print('{}->{} with {} history'.format(var_names[i], var_names[j],str(tau_max),))

            val = cmi.cmi_symb(array, X = X, Y = Y, Z = Z) # using FsATE
            print('FsATE_val: ',val)

            pval = cmi.symb_parallel_shuffles_significance(X, Y, Z, value = val)
            print('FsATE_pval: ', pval)

            P_corr, P_value = cmi.get_analytic_significance(array, X, Y, Z)

            print('partial_corr_val: ',P_corr,) 
            print('partial_corr_pval: ',P_value)
            if pval <= significance:
                if P_corr>0:
                    print('Positive dependent') 
                else:
                    print('Negative dependent')
            else:
                print('Independent')
