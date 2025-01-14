
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



# ############# COVID #################

FAMIs_covid_weekly = np.load('./Scaling_Results/FAMIs_covid_weekly.npy')
FAMIs_death_weekly = np.load('./Scaling_Results/FAMIs_death_weekly.npy')
FAMIs_fullvaccine_weekly = np.load('./Scaling_Results/FAMIs_fullvaccine_weekly.npy')



FAMIs_covid_dt = adf_detrend(FAMIs_covid_weekly)
FAMIs_death_dt = adf_detrend(FAMIs_death_weekly)
FAMIs_fullvaccine_dt = adf_detrend(FAMIs_fullvaccine_weekly)

N,T = FAMIs_covid_dt.shape

all = [FAMIs_covid_weekly,FAMIs_death_weekly,FAMIs_fullvaccine_weekly]
array_FAMI = np.zeros((N,len(all),T),dtype=float)
print('array_FAMI',array_FAMI.shape)

array_FAMI[:,0,:] = FAMIs_covid_dt
array_FAMI[:,1,:] = FAMIs_death_dt
array_FAMI[:,2,:] = FAMIs_fullvaccine_dt

var_names = ['$Covid$', '$Death$','$Vaccine$']


#########################
significance = 0.05
tau_max = 3 # time history
tau_lag = 3 # time lag

sig_samples = 10000; symbs = 4

array = array_FAMI
print('array_FAMI', var_names, array.shape)

cmi = CMI_estimator(array=array, sig_samples=sig_samples, symbs=symbs)
for i in range(array.shape[1]):
    for j in range(array.shape[1]):
        if i!=j:    
            z = list(range(array.shape[1])); z.remove(i); z.remove(j); k = z[0]
            Y = [(j, 0)]
            Z  = [(j, -t) for t in range(1, tau_max+1)]
            Z.append((k, -tau_lag))

            X = [(i, -tau_lag)]; 
            print('{}->{}|{} with {} lags {} history'.format(var_names[i], var_names[j], var_names[k],str(tau_lag),str(tau_max)))
            val = cmi.cmi_symb(array, X = X, Y = Y, Z = Z) # using FsACTE
            print('FsACTE_val: ',val)

            pval = cmi.symb_subset_parallel_shuffles_significance(X, Y, Z, value = val)
            print('FsACTE_pval: ', pval)
            P_corr, P_value = cmi.get_analytic_significance(array, X, Y, Z)
            print('partial_corr_val: ',P_corr,) 
            print('partial_corr_pval: ',P_value)
            if pval <= significance:
                if P_corr > 0:
                    print('Positive dependent') 
                else:
                    print('Negative dependent')
            else:
                print('Independent')
 