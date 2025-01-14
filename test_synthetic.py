

from FsATE_urban.CMI_estimator import CMI_estimator
import numpy as np
from scipy import stats, linalg
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.signal import detrend
import pandas as pd


def adf_detrend(data):
    # ADF Test on each column ccf3w

    # for name, column in df.iteritems():
        # Test for stationarity after detrending
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


def Regress(X,Y,Exposure1,Exposure2):
    Time_Steps, Ncities = X.shape
    Y_pd = pd.DataFrame({"Y":Y.flatten()})
    x_pd = pd.DataFrame({"X":X.flatten(), "n":np.tile(np.log(Exposure1),Time_Steps), "r":np.tile(np.log(Exposure2),Time_Steps)})
    X_pd = sm.add_constant(x_pd) 
    model = sm.OLS(Y_pd,X_pd)
    fit = model.fit()
    return fit


X2Y_pvals_regress = []
Y2X_pvals_regress = []
X2Y_pvals_PerCapita = []
Y2X_pvals_PerCapita = []
X2Y_pvals_FAMI = 0
Y2X_pvals_FAMI = 0
X2Y_pvals_SAMI = 0
Y2X_pvals_SAMI = 0
samples = 1 # number of realizations
N_list = [100] # number of cities
T = 100 # time steps



N_exp = np.linspace(0.5, 1.5, 6); R_exp = np.linspace(-.5, .5, 11)
R_exp_x = [0.5]; 

N_exp_X = 1.2; N_exp_Y = 0.3

sum_exp = np.linspace(0.5,1.5, 11); sum_exp_Y = np.linspace(0.5,1.5,11)

# r1 = 0.6; r2 = 0.8

ALL_X2Y_regress = np.zeros((5, 11, 11))
ALL_Y2X_regress = np.zeros((5, 11, 11))

ALL_X2Y_FsAMI = np.zeros((5, 11, 11))
ALL_Y2X_FsAMI = np.zeros((5, 11, 11))

for nn in range(len(N_list)):
    
    N = N_list[nn]
    
    for r1 in range(len(sum_exp)):
        for r2 in range(len(sum_exp)):
            
            R_exp_X = R_exp[r1]; R_exp_X = 0.6; sum_exp_X = N_exp_X + R_exp_X
            R_exp_Y = R_exp[r2]; R_exp_Y = 0.5; sum_exp_Y = N_exp_Y + R_exp_Y
            # R_exp_X = sum_exp[r1] - N_exp_X; R_exp_Y = sum_exp[r2] - N_exp_Y
            print('N = {}, sum_exp_X = {}, sum_exp_Y = {}, R_exp_X = {}, R_exp_Y = {}'.format(N, sum_exp_X, sum_exp_Y, R_exp_X, R_exp_Y))
            X2Y_pvals_regress = []
            Y2X_pvals_regress = []
            X2Y_pvals_PerCapita = []
            Y2X_pvals_PerCapita = []
            X2Y_pvals_FAMI = 0
            Y2X_pvals_FAMI = 0
            X2Y_pvals_SAMI = 0
            Y2X_pvals_SAMI = 0
            for sample in range(samples):
                
                ZETA_X = np.zeros([T,N])
                ZETA_Y = np.zeros([T,N])
                ZETA_X[0] = np.random.normal(0,1,N)
                ZETA_Y[0] = np.random.normal(0,1,N)
                for t in range(T-1):
                    for i in range(N):
                        ZETA_X[t+1,i] = 0.5*ZETA_X[t,i] + np.random.normal(0,1) + 2*ZETA_Y[t,i] 
                        ZETA_Y[t+1,i] = 0.6*ZETA_Y[t,i] + np.random.normal(0,1)

                ranks = np.linspace(1,N,N)
                expn_1 = 1; expn_2 = 1.5;
                # n = 10**5*(ranks**(-expn_1))
                # r = 10**3*(ranks**(-expn_2))
                pop_min_value = 10000;   
                pop_max_value = 1000000;  
                R_min_value = 100; 
                R_max_value = 100000; 

                n = (ranks**(-expn_1)) / max((ranks**(-expn_1))) * (pop_max_value - pop_min_value) + pop_min_value
                r = (ranks**(-expn_2)) / max((ranks**(-expn_2))) * (R_max_value - R_min_value) + R_min_value

                PerCapita_X = (n**(N_exp_X)*r**(R_exp_X)*np.exp(ZETA_X))/n
                PerCapita_Y = (n**(N_exp_Y)*r**(R_exp_Y)*np.exp(ZETA_Y))/n
                
                X2Y_pvals_regress.append(Regress(PerCapita_X[:-1,:],PerCapita_Y[1:,:],n,r).pvalues[0])
                Y2X_pvals_regress.append(Regress(PerCapita_Y[:-1,:],PerCapita_X[1:,:],n,r).pvalues[0])

                # ZETA_X = np.zeros((T,N),dtype=float)
                # ZETA_Y = np.zeros((T,N),dtype=float)

                XI_X = np.zeros((N,T),dtype=float)
                XI_Y = np.zeros((N,T),dtype=float)

                FAMI_X = np.zeros((N,T),dtype=float)
                FAMI_Y = np.zeros((N,T),dtype=float)


                def get_SAMI(target):
                    x = np.log(n)
                    y = np.log(target)    
                    X = sm.add_constant(x)
                    model = sm.OLS(y, X).fit()
                    # coef = model.params[1:].tolist()
                    # pval = model.pvalues[1:].tolist()
                    y_pred = model.params[0] + model.params[1]*x
                    residuals = y-y_pred
                    return residuals

                def get_FAMI(target):
                    x = np.zeros((N, 2))
                    x[:,0] = np.log(n)
                    x[:,1] = np.log(r)
                    # x[:,1] = np.log(R_att)
                    # x[:,2] = source
                    y = np.log(target)    
                    X = sm.add_constant(x)
                    model = sm.OLS(y, X).fit()
                    # coef = model.params[1:].tolist()
                    # pval = model.pvalues[1:].tolist()
                    y_pred = model.params[0]  + model.params[1]*x[:,0] + model.params[2]*x[:,1]
                    residuals = y-y_pred
                    return residuals    
                # array of FAMIs
                array_FAMI = np.zeros((N,2,T),dtype=float)
                array_FAMI[:,0,:] = ZETA_X.T
                array_FAMI[:,1,:] = ZETA_Y.T

                # array of per-capita
                array_PerCapita = np.zeros((N, 2, T), dtype=float)
                array_PerCapita[:,0,:] = PerCapita_X.T
                array_PerCapita[:,1,:] = PerCapita_Y.T

                var_names = ['$X$', '$Y$']

                # #########################
                significance = 0.05
                tau_max = 1 # time history
                tau_min = 1

                sig_samples = 1000
                array = array_FAMI
                # print('array_FAMI', var_names, array.shape)

                cmi = CMI_estimator(array=array, sig_samples=sig_samples, symbs=5)
                for i in range(array.shape[1]):
                    for j in range(array.shape[1]):
                        if i!=j:    
                            Y = [(j, 0)]
                            # optimal_tau,_ =  cmi.optimal_tau(array=array, j=j, tau_max=tau_max)
                            Z  = [(j, -t) for t in range(1, tau_max+1)]
                            X = [(i, -tau_max)]; 

                            val = cmi.cmi_symb(array, X = X, Y = Y, Z = Z)
                            pval = cmi.symb_parallel_shuffles_significance(X, Y, Z, value = val)

                            if j == 1:
                                if pval < 0.05:
                                    X2Y_pvals_FAMI = X2Y_pvals_FAMI + 1
                            else:
                                if pval < 0.05:
                                    Y2X_pvals_FAMI = Y2X_pvals_FAMI + 1

                            
                print((sample+1)/samples*100,'% complete FAMIs, ', 'X2Y FPR = ',100*X2Y_pvals_FAMI/(sample+1),'Y2X FPR = ',100*(Y2X_pvals_FAMI)/(sample+1))              

            X2Y_pvals_regress = np.array(X2Y_pvals_regress)
            Y2X_pvals_regress = np.array(Y2X_pvals_regress)



            print("Regression: X2Y",100*len(np.where(X2Y_pvals_regress<0.05)[0])/len(X2Y_pvals_regress))
            print("Regression: Y2X",100*len(np.where(Y2X_pvals_regress<0.05)[0])/len(Y2X_pvals_regress))


            print('FsAMIs: X2Y FPR = ',100*X2Y_pvals_FAMI/(sample+1))
            print('FsAMIs: Y2X FPR = ',100*(Y2X_pvals_FAMI)/(sample+1))       


            ALL_X2Y_regress[nn, r1, r2] = 100*len(np.where(X2Y_pvals_regress<0.05)[0])/len(X2Y_pvals_regress)  
            ALL_Y2X_regress[nn, r1, r2] = 100*len(np.where(Y2X_pvals_regress<0.05)[0])/len(Y2X_pvals_regress)      

            ALL_X2Y_FsAMI[nn, r1, r2] = 100*X2Y_pvals_FAMI/(sample+1)
            ALL_Y2X_FsAMI[nn, r1, r2] = 100*Y2X_pvals_FAMI/(sample+1)