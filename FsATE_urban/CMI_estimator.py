from __future__ import print_function
from numba import jit
import warnings
import numpy as np
# from copy import deepcopy
from scipy.stats.contingency import crosstab
# from itertools import combinations
from scipy.stats import pearsonr
from scipy import spatial, special
import math
# from data_processing import DataFrame
from scipy import stats, linalg
import sys


class CMI_estimator():                                                                                                                                                                                                                                                              
    def __init__(self,
                 array,
                 symbs = 4,
                 sig_samples=1000,
                 shuffle_neighbors = 10,
                 sig_blocklength = None,
                 time_lag = 1,
                 transform='standardize',
                 workers=-1):
        self.N,self.dim,self.T = array.shape
        # self.constraint = constraint
        self.array = array
        # self.n_symbs = n_symbs+1
        # self.const = np.unique(constraint,axis=0)
        self.num = self.N*self.T
        self.seed = 5
        self.sig_samples = sig_samples
        self.symbs = symbs+1
        self.time_lag = time_lag
        self.shuffle_neighbors = shuffle_neighbors
        self.sig_blocklength = sig_blocklength

        self.workers = workers
        self.random_state = np.random.default_rng(self.seed)
        self.transform = transform



    def binning(self, values, percentiles):
        # Flatten the array to create a 1D array for binning
        flat_values = values.flatten()
        # Digitize the values into bins
        binned_indices = np.digitize(flat_values, percentiles, right=True)
        symbolized_array = binned_indices.reshape(values.shape)
        # print(np.unique(symbolized_array))
        return symbolized_array


    def joint_entropy(self, p): 
        """ 
        Parameter
        ------
        p: joint probability of (X,Y,Z,...) or probability of X
        """
        T = np.sum(p)
        p = p[p!=0]/T
        
        return -np.sum(p*np.log(p))
    
    # def array_symb(self, array):
    #     for d in range(array.shape[0]):
    #         percentile = [np.percentile(array[d], 25),np.percentile(array[d], 50),np.percentile(array[d], 75),]
    #         # percentile = [np.percentile(array[d], 20),np.percentile(array[d], 40),np.percentile(array[d], 60),np.percentile(array[d], 80),]
    #         array[d] = self.symbolize(array[d], percentile)
    #     return array
    

    def optimal_tau(self, array, j ,tau_max):

        """ 
        Parameters
        ------
        ZETA_symb: symbolized array
        tau_max: maximum time lag

        Returns
        -------
        cond_entropy : tuple
            (optimal lag, conditional entropy)
        """

        N, dim, T = array.shape

        # array_symb = np.zeros((N, dim, T), dtype=float)
        # for d in range(dim):
        #     # median_value =np.median(array[:,d])
        #     percentile = [np.percentile(array[:,d], 25),np.percentile(array[:,d], 50),np.percentile(array[:,d], 75),]
        #     for n in range(N):
        #         array_symb[n, d] = self.symbolize(array[n, d], percentile)
        y_array = np.c_[array[:, j, tau_max: ].ravel()].T
        y_symb = self.symbolize_array(y_array)
        
        # y_array = np.c_[array_symb[:, j, tau_max: ].ravel()].T

        # print(yz_array.shape, z_array.shape, y_array.shape)
        # print(y_symb)
        hist_y = crosstab(*y_symb).count.flatten();hy = self.joint_entropy(hist_y)
        # print(hy)
        cond_entropy = {}
        tau = 1
        while tau <= tau_max:
            yz_array = np.zeros((tau+1, N*(T-tau)), dtype=float)
            z_array = np.zeros((tau, N*(T-tau)), dtype=float)
            for tt in range(tau+1):
                if tt == 0 :
                    yz_array[tt] = array[:, j, tau:].ravel()
                elif tt == tau-1:
                    yz_array[tt] = array[:, j, : -tau].ravel()
                else:
                    yz_array[tt] = array[:, j, tau-tt:-tt].ravel()
            yz_symb = self.symbolize_array(yz_array)

            z_array = yz_array[1:];z_symb = self.symbolize_array(z_array)

            hist_yz = crosstab(*yz_symb).count.flatten() 
            hist_z = crosstab(*z_symb).count.flatten()

            hyz = self.joint_entropy(hist_yz)
            hz = self.joint_entropy(hist_z)
            val = hyz - hz  ## entropy H(Yt|Yt-1,Yt_2,...,Yt_tau)

            cond_entropy[tau] = val
            optimal_entropy = tuple((tau, val))
            if tau > 1 and val < cond_entropy[tau-1]:
                optimal_entropy = tuple((tau, val))
                # print(optimal_entropy, cond_entropy[tau-1] - val, 0.02*hy)
                # if cond_entropy[tau-1] - val < 0.02*hy:
                #     optimal_entropy = tuple((tau-1, cond_entropy[tau-1]))
                    # print('no pss')

                    # break
            # elif tau > 1 and cond_entropy[tau-1] - val < 0.02*hy:
            #     # optimal_entropy = tuple((tau-1, cond_entropy[tau-1]))
            #     print(optimal_entropy)
            #     break
                
            tau += 1
        return optimal_entropy     
        
    def _array_reshape(self, array, X, Y, Z):
        """Returns time shifted array and indices.

        Parameters
        ----------
        array: unshuffled (N, dim, T)

        X,Y,Z: indices of target, int or list, i.e., X = 1 or [1, 2, ...]

        Returns
        -------
        xyz : array of ints
            identifier array of shape (dim,).

        array : array-like (N*(T-time_lag), dim)
            indexed array with time shift.
        """
        N, dim, T = array.shape
        x, xt = X[0]; y, yt = Y[0]
        

        if len(Z)>0:
            z = list(list(zip(*Z))[0]); zt = list(list(zip(*Z))[1])
            t_min = min([xt]+zt)

            if abs(xt)<1:array_shift_X = array[:,x,abs(t_min-xt):].ravel() # it allows to test independence on variables without time lags (tau = 0)
            else:array_shift_X = array[:,x,abs(t_min-xt):xt].ravel()

            array_shift_Y = array[:,y,abs(t_min-yt):].ravel()
            Z_array = np.array([[array[n,z[i],abs(t_min-zt[i]):zt[i]] for i in range(len(z))] for n in range(N)])
            # print(Z_array.shape)
            # print(len(z),xt,zt, t_min)
            if len(z)>1:
                array_shift_Z = [Z_array[:,i].ravel() for i in range(len(z))]
                array_shift_Z = np.array(array_shift_Z).reshape(-1,len(z))
            else:
                array_shift_Z = Z_array.ravel()

            array_shift = np.c_[array_shift_X, array_shift_Y, list(array_shift_Z)]        
        else: # no condition Z = []
            t_min = min([xt])
            if abs(xt)<1:array_shift_X = array[:,x,abs(t_min-xt):].ravel()
            else:array_shift_X = array[:,x,abs(t_min-xt):xt].ravel()
            array_shift_Y = array[:,y,abs(t_min-yt):].ravel()
            array_shift = np.c_[array_shift_X, array_shift_Y]
        xyz = np.array([0, 1]+list(np.ones(len(Z))*2))

        return array_shift, xyz
    

    def symbolize_array(self, array,):
        _, dim = array.shape
        points = np.linspace(0, 100, self.symbs)[1:-1]
        for d in range(dim):
            percentile = [np.percentile(array[:,d], p) for p in points]
            array[:, d] = self.binning(array[:,d], percentile)
        return array
    


    def cmi_symb(self, array, X, Y, Z):
        """ 
        Parameters
        ------
        array: array-like, (N, dim, T)

        Returns
        ------
        val: estimated conditional mutual information

        """

        array_shift, xyz = self._array_reshape(array, X, Y, Z)
        T, dim = array_shift.shape
        # print(array_shift.shape)
        # # array_symb = np.zeros((T, dim), dtype=float)

        # Binning array
        array_shift = self.symbolize_array(array_shift)

        # print(array_shift)
        x_indice = np.where(xyz==0)[0]
        y_indice = np.where(xyz==1)[0]
        z_indices = np.where(xyz==2)[0]


        
        if len(z_indices) > 0:

            # print(len(z_indices))
            xz_array = array_shift[:, np.concatenate((x_indice, z_indices))].T
            yz_array = array_shift[:, np.concatenate((y_indice, z_indices))].T
            # xyz_array = array_shift[:, np.concatenate((x_indice, y_indice, z_indices))].T
            xyz_array = array_shift.T
            z_array = array_shift[:, z_indices].T

            hist_xz = crosstab(*xz_array).count.flatten(); hxz = self.joint_entropy(hist_xz)
            hist_yz = crosstab(*yz_array).count.flatten(); hyz = self.joint_entropy(hist_yz)
            hist_xyz = crosstab(*xyz_array).count.flatten(); hxyz = self.joint_entropy(hist_xyz)
            hist_z = crosstab(*z_array).count.flatten(); hz = self.joint_entropy(hist_z)
            # print(hxz, hyz, hxyz, hz)
            val = hxz + hyz - hxyz - hz
        

        
        else:

            x_array = array_shift[:, x_indice].T
            y_array = array_shift[:, y_indice].T
            xy_array = array_shift[:, np.concatenate((x_indice, y_indice))].T

            hist_x = crosstab(*x_array).count.flatten()
            hist_y = crosstab(*y_array).count.flatten()
            hist_xy = crosstab(*xy_array).count.flatten()

            hx = self.joint_entropy(hist_x)
            hy = self.joint_entropy(hist_y)
            hxy = self.joint_entropy(hist_xy)

            val = hx + hy - hxy 

        return val
        
    

    def _cmi_symb(self, array, X,Y,Z):
        """ 
        Parameters
        ------
        array: array-like, (N, dim, T)

        Returns
        ------
        val: estimated conditional mutual information

        """
        N, dim, T = array.shape

        array_shift, xyz = self._array_reshape(array, X, Y, Z)
        T, dim = array_shift.shape
        # print(array_shift.shape)
        # # array_symb = np.zeros((T, dim), dtype=float)
        
        array_shift = self.symbolize_array(array_shift)

        # print(array_shift)
        x_indice = np.where(xyz==0)[0]
        y_indice = np.where(xyz==1)[0]
        z_indices = np.where(xyz==2)[0]

        
        if len(z_indices) > 0:

            # print(len(z_indices))
            xz_array = array_shift[:, np.concatenate((x_indice, z_indices))].T
            yz_array = array_shift[:, np.concatenate((y_indice, z_indices))].T
            # xyz_array = array_shift[:, np.concatenate((x_indice, y_indice, z_indices))].T
            xyz_array = array_shift.T
            z_array = array_shift[:, z_indices].T

            hist_xz = crosstab(*xz_array).count.flatten(); hxz = self.joint_entropy(hist_xz)
            hist_yz = crosstab(*yz_array).count.flatten(); hyz = self.joint_entropy(hist_yz)
            hist_xyz = crosstab(*xyz_array).count.flatten(); hxyz = self.joint_entropy(hist_xyz)
            hist_z = crosstab(*z_array).count.flatten(); hz = self.joint_entropy(hist_z)
            # print(hxz, hyz, hxyz, hz)
            val = hxz + hyz - hxyz - hz
        

        
        else:

            x_array = array_shift[:, x_indice].T
            y_array = array_shift[:, y_indice].T
            xy_array = array_shift[:, np.concatenate((x_indice, y_indice))].T

            hist_x = crosstab(*x_array).count.flatten()
            hist_y = crosstab(*y_array).count.flatten()
            hist_xy = crosstab(*xy_array).count.flatten()

            hx = self.joint_entropy(hist_x)
            hy = self.joint_entropy(hist_y)
            hxy = self.joint_entropy(hist_xy)

            val = hx + hy - hxy 

        return val


    def symb_parallel_shuffles_significance(self, X, Y,Z, value):

        random_seeds = np.random.default_rng(self.seed).integers(np.iinfo(np.int32).max, size=(self.sig_samples))
        null_dist = np.zeros(self.sig_samples)


        for i, seed in enumerate(random_seeds):
            shuffle_val = self.symb_parallel_shuffles(X, Y, Z, seed=seed)
            # print('shuffled value',shuffle_val)
            null_dist[i] = shuffle_val
        
        pval = (null_dist >= value).mean()

        # print('95% quantile: ',np.percentile(null_dist, 95))

        return pval

    

    def symb_parallel_shuffles(self, X,Y,Z, seed=None):
        """Returns shuffled array over first column

        Parameters
        ----------
        array : array of XYZ 
            XYZ is array [Xt,Yt1,Yt].T

        Returns
        -------
        value : array-like
            array with the first column (Xt) shuffled.
        
        """
        array = self.array
        x, xt = X[0]; y, _ = Y[0]
        # z = list(list(zip(*Z))[0]); zt = list(list(zip(*Z))[1])
        # x = np.where(xyz==0)[0]
        N, dim, T = array.shape
        rng = np.random.default_rng(seed=None)
        array_shuffled = np.copy(array)

        for n in range(N):
        #     for i in z:
            order = rng.permutation(T).astype('int32')
            array_shuffled[n,x] = array_shuffled[n,x,order] # Shuffle X variable
            # order2 = rng.permutation(T).astype('int32')
            # array_shuffled[n, y, :] = array_shuffled[n, y, order2] # Shuffle X variable
        return self._cmi_symb(array_shuffled,X,Y,Z)
    


    def symb_subset_parallel_shuffles_significance(self, X, Y,Z, value):

        random_seeds = np.random.default_rng(self.seed).integers(np.iinfo(np.int32).max, size=(self.sig_samples))
        null_dist = np.zeros(self.sig_samples)


        for i, seed in enumerate(random_seeds):
            shuffle_val = self.symb_subset_parallel_shuffles(X, Y, Z, seed=seed)
            # print('shuffled value',shuffle_val)
            null_dist[i] = shuffle_val
        
        pval = (null_dist >= value).mean()

        print('95% quantile: ',np.percentile(null_dist, 95))

        return pval

    

    def symb_subset_parallel_shuffles(self, X,Y,Z, seed=None):
        """Returns shuffled array over first column

        Parameters
        ----------
        array : array of XYZ 
            XYZ is array [Xt,Yt1,Yt].T

        Returns
        -------
        value : array-like
            array with the first column (Xt) shuffled while preserving the association between (Yt, Zt).
        
        """
        array = self.array
        x, xt = X[0]; y, _ = Y[0]
        # z = list(list(zip(*Z))[0]); zt = list(list(zip(*Z))[1])
        # x = np.where(xyz==0)[0]
        N, dim, T = array.shape

        rng = np.random.default_rng(seed=None)

        array_shift, xyz = self._array_reshape(array, X, Y, Z)
        array_symb = self.symbolize_array(array_shift)
        T_new, dim = array_symb.shape

        array_symb = array_symb.reshape(N, dim, int(T_new/N))


        array_shuffled = np.copy(array_symb)


        x_indice = np.where(xyz==0)[0]
        y_indice = np.where(xyz==1)[0]
        z_indice = np.where(xyz==2)[0]

        for n in range(N):
            tuples = array_symb[n, np.concatenate((y_indice, z_indice[-1:]))].T
            unique_tuples, inverse_indices = np.unique(tuples, axis=0, return_inverse=True)

            for i, ut in enumerate(unique_tuples):
                locations = np.where(inverse_indices == i)[0] 
                order = rng.permutation(locations).astype('int32')
                array_shuffled[n, x_indice, locations] = array_shuffled[n, x_indice, order]
    
        return self._cmi_symb_subset(array_shuffled,X,Y,Z)
    
    def _cmi_symb_subset(self, array, X,Y,Z):
        """ 

        Parameters
        ----------
        array: symbolic array (N, dim, T)
        
        Returns
        -------
        val: surrogate (conditional) transfer entropy
        """

        N, dim, T = array.shape

        array_shift, xyz = self._array_reshape(array, X, Y, Z)
        T, dim = array_shift.shape

        x_indice = np.where(xyz==0)[0]
        y_indice = np.where(xyz==1)[0]
        z_indices = np.where(xyz==2)[0]

        
        if len(z_indices) > 0:

            # print(len(z_indices))
            xz_array = array_shift[:, np.concatenate((x_indice, z_indices))].T
            yz_array = array_shift[:, np.concatenate((y_indice, z_indices))].T
            # xyz_array = array_shift[:, np.concatenate((x_indice, y_indice, z_indices))].T
            xyz_array = array_shift.T
            z_array = array_shift[:, z_indices].T

            hist_xz = crosstab(*xz_array).count.flatten(); hxz = self.joint_entropy(hist_xz)
            hist_yz = crosstab(*yz_array).count.flatten(); hyz = self.joint_entropy(hist_yz)
            hist_xyz = crosstab(*xyz_array).count.flatten(); hxyz = self.joint_entropy(hist_xyz)
            hist_z = crosstab(*z_array).count.flatten(); hz = self.joint_entropy(hist_z)
            # print(hxz, hyz, hxyz, hz)
            val = hxz + hyz - hxyz - hz
        

        
        else:

            x_array = array_shift[:, x_indice].T
            y_array = array_shift[:, y_indice].T
            xy_array = array_shift[:, np.concatenate((x_indice, y_indice))].T

            hist_x = crosstab(*x_array).count.flatten()
            hist_y = crosstab(*y_array).count.flatten()
            hist_xy = crosstab(*xy_array).count.flatten()

            hx = self.joint_entropy(hist_x)
            hy = self.joint_entropy(hist_y)
            hxy = self.joint_entropy(hist_xy)

            val = hx + hy - hxy 

        return val

    ############# CMI_KNN ##################

    def _nearest_neighbors(self,array,xyz, knn):
        """Returns nearest neighbors according to Frenzel and Pompe (2007).

        Retrieves the distances eps to the k-th nearest neighbors for every
        sample in joint space XYZ and returns the numbers of nearest neighbors
        within eps in subspaces Z, XZ, YZ.

        Parameters
        ----------
        array : array-like, only two dimensions (X and Y)
            After time shift, data array with X_t, Y_t+1, Y_t in rows and observations in columns
            array of shape (N*(T-time_lag),(x_t,y_t1,(y_t+z_t)),)
        

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        knn : int or float
            Number of nearest-neighbors which determines the size of hyper-cubes
            around each (high-dimensional) sample point. If smaller than 1, this
            is computed as a fraction of T, hence knn=knn*T. For knn larger or
            equal to 1, this is the absolute number.

        Returns
        -------
        k_xz, k_yz, k_z : tuple of arrays of shape (T,)
            Nearest neighbors in subspaces.
        """
        array = array.astype(np.float64).T
        
        dim, T = array.shape
        # self.N,self.dim,self.T = array.shape
        # Add noise to destroy ties...

        array += (1E-6 * array.std(axis=1).reshape(dim, 1)
                  * self.random_state.random((array.shape[0], array.shape[1])))
        # array += (1E-6 
        #           * self.random_state.random((array.shape[0], array.shape[1])))
        # array = array.T
        if self.transform == 'standardize':
            # Standardize
            array = array.astype(np.float64)
            array -= array.mean(axis=1).reshape(dim, 1)
            std = array.std(axis=1)
            for i in range(dim):
                if std[i] != 0.:
                    array[i] /= std[i] 
                # array /= array.std(axis=1).reshape(dim, 1)
                # FIXME: If the time series is constant, return nan rather than
                # raising Exception
                # if np.any(std == 0.) and self.verbosity > 0:
                #     warnings.warn("Possibly constant array!")
                #     # raise ValueError("nans after standardizing, "
                #     #                  "possibly constant array!")
        elif self.transform == 'uniform':
            array = self._trafo2uniform(array)
        elif self.transform == 'ranks':
            array = array.argsort(axis=1).argsort(axis=1).astype(np.float64)

        array = array.T
        tree_xyz = spatial.cKDTree(array) 
        epsarray = tree_xyz.query(array, k=[knn+1], p=np.inf,
                                  eps=0., workers=self.workers)[0][:, 0].astype(np.float64) 
        
        # To search neighbors < eps
        epsarray = np.multiply(epsarray, 0.99999)

        # Subsample indices
        x_indices = np.where(xyz == 0)[0]
        y_indices = np.where(xyz == 1)[0]
        z_indices = np.where(xyz == 2)[0]

        # Find nearest neighbors in subspaces
        xz = array[:, np.concatenate((x_indices, z_indices))]
        tree_xz = spatial.cKDTree(xz)
        k_xz = tree_xz.query_ball_point(xz, r=epsarray, eps=0., p=np.inf, workers=self.workers, return_length=True)

        yz = array[:, np.concatenate((y_indices, z_indices))]
        tree_yz = spatial.cKDTree(yz)
        k_yz = tree_yz.query_ball_point(yz, r=epsarray, eps=0., p=np.inf, workers=self.workers, return_length=True)

        if len(z_indices) > 0:
            z = array[:, z_indices]
            tree_z = spatial.cKDTree(z)
            k_z = tree_z.query_ball_point(z, r=epsarray, eps=0., p=np.inf, workers=self.workers, return_length=True)
        else:
            # Number of neighbors is T when z is empty.
            k_z = np.full(T, T, dtype=np.float64)

        # x = 0; y = 1; 
        # z = list(range(dim)); z.remove(x); z.remove(y)
        # xz = [x]+z; yz = [y]+z
        
        # XZ = array[:,xz] 
        # tree_xz= spatial.cKDTree(XZ)
        # k_xz = tree_xz.query_ball_point(XZ,r=epsarray,eps=0.,p=np.inf,workers=self.workers,return_length=True)

        # YZ = array[:,yz]
        # tree_yz = spatial.cKDTree(YZ)
        # k_yz = tree_yz.query_ball_point(YZ,r=epsarray,eps=0.,p=np.inf,workers=self.workers,return_length=True)

        # if len(z) > 1:
        #     Z = array[:,z]
        #     tree_z = spatial.cKDTree(Z)
        #     k_z = tree_z.query_ball_point(Z,r=epsarray,eps=0.,p=np.inf,workers=self.workers,return_length=True)
        # else:
        #     # Number of neighbors is T when z is empty.
        #     k_z = np.full(T,T, dtype=np.float64)
            

        return k_xz, k_yz, k_z
    

    def independence_measure(self, array, X, Y, Z):
        """Returns CMI estimate as described in Frenzel and Pompe PRL (2007).

        Parameters
        ----------
        array: shuffled or unshuffled (N, dim, T)

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        X,Y,Z: indices and time lags of target, list of tuple(s), i.e., X = [(0, -1)]

        Returns
        -------
        XYZ : array with shape (N*(T-t_min), dim)

        value : float
            Conditional mutual information estimate.
        """
        array_shift, xyz = self._array_reshape(array, X, Y, Z)

        T, dim = array_shift.shape

        if self.knn < 1:
            knn_here = max(1, int(self.knn*T))
        else:
            knn_here = max(1, int(self.knn))
        
        k_xz, k_yz, k_z = self._nearest_neighbors(array=array_shift,
                                                  xyz=xyz,
                                                  knn=knn_here)
        if len(Z)>0:
            value = special.digamma(knn_here) - (special.digamma(k_xz) +
                                            special.digamma(k_yz) -
                                            special.digamma(k_z)).mean()
        else:
        
            value = special.digamma(knn_here) + special.digamma(T) - (special.digamma(k_xz) +
                                           special.digamma(k_yz)).mean()

        return value
    

    def _independence_measure(self, array, xyz):

        z_indices = np.where(xyz==2)[0]

        T, dim = array.shape

        if self.knn < 1:
            knn_here = max(1, int(self.knn*T))
        else:
            knn_here = max(1, int(self.knn))
        
        k_xz, k_yz, k_z = self._nearest_neighbors(array=array,
                                                  xyz=xyz,
                                                  knn=knn_here)
        if len(z_indices)>0:
            value = special.digamma(knn_here) - (special.digamma(k_xz) +
                                            special.digamma(k_yz) -
                                            special.digamma(k_z)).mean()
        else:
        
            value = special.digamma(knn_here) + special.digamma(T) - (special.digamma(k_xz) +
                                           special.digamma(k_yz)).mean()

        return value
    


    # def parallel_shuffles(self, X, Y, Z, seed=None):
    #     """Returns shuffled array over first column

    #     Parameters
    #     ----------
    #     array : array of XYZ 
    #         XYZ is array [Xt,Yt1,Yt].T

    #     Returns
    #     -------
    #     value : array-like
    #         array with the first column (Xt) shuffled.
        
    #     """
    #     array = self.array
    #     x, xt = X[0]; y, _ = Y[0]
        
    #     N, dim, T = array.shape
    #     rng = np.random.default_rng(seed=None)
    #     array_shuffled = np.copy(array)
    #     for n in range(N):
    #         order = rng.permutation(T).astype('int32')
    #         array_shuffled[n, x, :] = array_shuffled[n, x, order] # Shuffle X variable

    #     return self.independence_measure(array_shuffled, X,Y,Z)
  

    def parallel_shuffles_significance(self, X, Y ,Z, value):
        # x,y,z = X,Y,Z
        # while r < self.dim-1:
        #     for l in list(combinations(range(self.dim),2)):
        # value = self.independence_measure(self.array, X,Y,Z)
        # print('real',value)
        random_seeds = np.random.default_rng(self.seed).integers(np.iinfo(np.int32).max, size=(self.sig_samples))
        null_dist = np.zeros(self.sig_samples)

        for i, seed in enumerate(random_seeds):
            shuffled_value = self.knn_parallel_shuffles(X,Y,Z, seed=seed)
            # print('shuffled value',shuffled_value)
            # if shuffled_value > value:
            #     print('LARGE')
            null_dist[i] = shuffled_value
        
        pval = (null_dist >= value).mean()

        return pval
    
    def knn_parallel_shuffles(self, X, Y, Z, seed=None):
        """Returns shuffled array over first column

        Parameters
        ----------
        array : array of XYZ 
            XYZ is array [Xt,Yt1,Yt].T

        Returns
        -------
        value : array-like
            array with the first column (Xt) shuffled.
        
        """
        array = self.array
        array_shift = array
        # array_shift, xyz = self._array_reshape(array, X, Y, Z)
        x, xt = X[0]; y, _ = Y[0]
        
        N,_,T = array_shift.shape
        rng = np.random.default_rng(seed=None)
        array_shuffled = np.copy(array_shift)
        
        order = rng.permutation(T).astype('int32')
        # array_shuffled[:, 0] = array_shuffled[order, x] # Shuffle X variable
        for n in range(N):
            order = rng.permutation(T).astype('int32')
            array_shuffled[n, x, :] = array_shuffled[n, x, order] # Shuffle X variable
        return self.independence_measure(array_shuffled, X,Y,Z)
    
    def knn_shuffle_significance(self, X, Y, Z, value,
                                 return_null_dist=False):
        """Returns p-value for nearest-neighbor shuffle significance test.

        For non-empty Z, overwrites get_shuffle_significance from the parent
        class  which is a block shuffle test, which does not preserve
        dependencies of X and Y with Z. Here the parameter shuffle_neighbors is
        used to permute only those values :math:`x_i` and :math:`x_j` for which
        :math:`z_j` is among the nearest niehgbors of :math:`z_i`. If Z is
        empty, the block-shuffle test is used.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        value : number
            Value of test statistic for unshuffled estimate.

        Returns
        -------
        pval : float
            p-value
        """
        array_shift,xyz = self._array_reshape(self.array, X, Y, Z)
        array = array_shift.T
        dim, T = array.shape

        # Skip shuffle test if value is above threshold
        # if value > self.minimum threshold:
        #     if return_null_dist:
        #         return 0., None11


        
        #     else:
        #         return 0.

        # max_neighbors = max(1, int(max_neighbor_ratio*T))
        x_indices = np.where(xyz == 0)[0]
        y_indices = np.where(xyz == 0)[0]
        z_indices = np.where(xyz == 2)[0]

        if len(z_indices) > 0 and self.shuffle_neighbors < T:
            # if self.verbosity > 2:
            #     print("            nearest-neighbor shuffle significance "
            #           "test with n = %d and %d surrogates" % (
            #           self.shuffle_neighbors, self.sig_samples))

            # Get nearest neighbors around each sample point in Z
            z_array = np.fastCopyAndTranspose(array[z_indices, :])
            tree_xyz = spatial.cKDTree(z_array)
            neighbors = tree_xyz.query(z_array,
                                       k=self.shuffle_neighbors,
                                       p=np.inf,
                                       eps=0.)[1].astype(np.int32)

            null_dist = np.zeros(self.sig_samples)
            for sam in range(self.sig_samples):

                # Generate random order in which to go through indices loop in
                # next step
                order = self.random_state.permutation(T).astype(np.int32)

                # Shuffle neighbor indices for each sample index
                for i in range(len(neighbors)):
                    self.random_state.shuffle(neighbors[i])
                # neighbors = self.random_state.permuted(neighbors, axis=1)
                
                # Select a series of neighbor indices that contains as few as
                # possible duplicates
                restricted_permutation = self.get_restricted_permutation(
                        T=T,
                        shuffle_neighbors=self.shuffle_neighbors,
                        neighbors=neighbors,
                        order=order)

                array_shuffled = np.copy(array)
                for i in x_indices:
                    array_shuffled[i] = array[i, restricted_permutation]
                shuffled_value = self._independence_measure(array_shuffled.T,
                                                             xyz)
                null_dist[sam] = shuffled_value
                print(shuffled_value)
                if shuffled_value > value:
                    print('large')

        # else:
        #     null_dist = \
        #             self._get_shuffle_dist(array, xyz,
        #                                    self.get_dependence_measure,
        #                                    sig_samples=self.sig_samples,
        #                                    sig_blocklength=self.sig_blocklength,
        #                                    verbosity=self.verbosity)

        pval = (null_dist >= value).mean()

        if return_null_dist:
            # Sort
            null_dist.sort()
            return pval, null_dist
        return pval



    def reshape_parallel_shuffles_significance(self, X, Y ,Z, value):
        xyz = np.array([0, 1]+list(np.ones(len(Z))*2))
        # while r < self.dim-1:
        #     for l in list(combinations(range(self.dim),2)):
        # value = self.independence_measure(self.array, X,Y,Z)
        print('real',value)
        random_seeds = np.random.default_rng(self.seed).integers(np.iinfo(np.int32).max, size=(self.sig_samples))
        null_dist = np.zeros(self.sig_samples)

        for i, seed in enumerate(random_seeds):
            print('shuffled value',self.reshape_parallel_shuffles(X,Y,Z, seed=seed))
            null_dist[i] = self.reshape_parallel_shuffles(X,Y,Z, seed=seed)
        
        pval = (null_dist >= value).mean()

        return pval
    
    

    def _trafo2uniform(self, x):
        """Transforms input array to uniform marginals.

        Assumes x.shape = (dim, T)

        Parameters
        ----------
        x : array-like
            Input array.

        Returns
        -------
        u : array-like
            array with uniform marginals.
        """

        def trafo(xi):
            xisorted = np.sort(xi)
            yi = np.linspace(1. / len(xi), 1, len(xi))
            return np.interp(xi, xisorted, yi)

        if np.ndim(x) == 1:
            u = trafo(x)
        else:
            u = np.empty(x.shape)
            for i in range(x.shape[0]):
                u[i] = trafo(x[i])
        return u
    
    
    def partial_corr(self, array, X, Y, Z):
        """
        Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
        for the remaining variables in C.
        Parameters
        ----------
        C : array-like, shape (n, p)
            Array with the different variables. Each column of C is taken as a variable
        Returns
        -------
        P : array-like, shape (p, p)
            P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
            for the remaining variables in C.
        """
        
        array_shift, _ = self._array_reshape(array, X=X, Y=Y, Z=Z)

        C = np.asarray(array_shift)
        p = C.shape[1]
        P_corr = np.zeros((p, p), dtype=np.float)
        P_value = np.zeros((p, p), dtype=np.float)
        for i in range(p):
            P_corr[i, i] = 1
            for j in range(i+1, p):
                idx = np.ones(p, dtype=np.bool)
                idx[i] = False
                idx[j] = False
                beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
                beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

                res_j = C[:, j] - C[:, idx].dot( beta_i)
                res_i = C[:, i] - C[:, idx].dot(beta_j)
                
                corr = stats.pearsonr(res_i, res_j)[0]
                pval = stats.pearsonr(res_i, res_j)[1]
                P_corr[i, j] = corr
                P_corr[j, i] = corr
                P_value[i, j] = pval
                P_value[j, i] = pval
            
        return P_corr[0,1], P_value[0,1]


    def pearson_corr(self, array, X, Y, Z):
        """
        Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
        for the remaining variables in C.
        Parameters
        ----------
        C : array-like, shape (n, p)
            Array with the different variables. Each column of C is taken as a variable
        Returns
        -------
        P : array-like, shape (p, p)
            P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
            for the remaining variables in C.
        """
        
        array_shift, xyz = self._array_reshape(self, array, X=X, Y=Y, Z=Z)
        T,_ = array_shift.shape
        dim_x = np.where(xyz==0)[0]
        dim_y = np.where(xyz==1)[0]
        print(array_shift[:,dim_x].shape)
        # print(array_shift[:,dim_y])
        val, pval = pearsonr(array_shift[:,dim_x], array_shift[:,dim_y])

        return val, pval

    @jit(forceobj=True)
    def get_restricted_permutation(self, T, shuffle_neighbors, neighbors, order):

        restricted_permutation = np.zeros(T, dtype=np.int32)
        used = np.array([], dtype=np.int32)

        for sample_index in order:
            m = 0
            use = neighbors[sample_index, m]

            while ((use in used) and (m < shuffle_neighbors - 1)):
                m += 1
                use = neighbors[sample_index, m]

            restricted_permutation[sample_index] = use
            used = np.append(used, use)

        return restricted_permutation


    def _get_acf(self,series, max_lag=None):
        """Returns autocorrelation function.

        Parameters
        ----------
        series : 1D-array
            data series to compute autocorrelation from

        max_lag : int, optional (default: None)
            maximum lag for autocorrelation function. If None is passed, 10% of
            the data series length are used.

        Returns
        -------
        autocorr : array of shape (max_lag + 1,)
            Autocorrelation function.
        """
        # Set the default max lag
        if max_lag is None:
            max_lag = int(max(5, 0.1*len(series)))
        # Initialize the result
        autocorr = np.ones(max_lag + 1)
        # Iterate over possible lags
        for lag in range(1, max_lag + 1):
            # Set the values
            y1_vals = series[lag:]
            y2_vals = series[:len(series) - lag]
            # Calculate the autocorrelation

            autocorr[lag] = np.corrcoef(y1_vals, y2_vals, ddof=0)[0, 1]
        return autocorr

    def _get_block_length(self,array, X,Y,Z, mode):
        """Returns optimal block length for significance and confidence tests.

        Determine block length using approach in Mader (2013) [Eq. (6)] which
        improves the method of Peifer (2005) with non-overlapping blocks In
        case of multidimensional X, the max is used. Further details in [1]_.
        Two modes are available. For mode='significance', only the indices
        corresponding to X are shuffled in array. For mode='confidence' all
        variables are jointly shuffled. If the autocorrelation curve fit fails,
        a block length of 5% of T is used. The block length is limited to a
        maximum of 10% of T.

        Mader et al., Journal of Neuroscience Methods,
        Volume 219, Issue 2, 15 October 2013, Pages 285-291

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        mode : str
            Which mode to use.

        Returns
        -------
        block_len : int
            Optimal block length.
        """
        # Inject a dependency on siganal, optimize
        from scipy import signal, optimize
        # Get the shape of the array
        x,xt = X[0]
        dim, T = array.shape
        # Initiailize the indices
        indices = range(dim)
        if mode == 'significance':
            indices = [x]

        # Maximum lag for autocov estimation
        max_lag = int(0.1*T)
        # Define the function to optimize against
        def func(x_vals, a_const, decay):
            return a_const * decay**x_vals

        # Calculate the block length
        block_len = 1
        for i in indices:
            # Get decay rate of envelope of autocorrelation functions
            # via hilbert trafo
                # print(_get_acf(series=array[n,i], max_lag=max_lag).shape)
            autocov = self._get_acf(series=array[i], max_lag=max_lag)
            autocov[0] = 1.
            hilbert = np.abs(signal.hilbert(autocov))
            # Try to fit the curve
            try:
                popt, _ = optimize.curve_fit(
                    f=func,
                    xdata=np.arange(0, max_lag+1),
                    ydata=hilbert,
                )
                phi = popt[1]
                # Formula assuming non-overlapping blocks
                l_opt = (4. * T * (phi / (1. - phi) + phi**2 / (1. - phi)**2)**2
                            / (1. + 2. * phi / (1. - phi))**2)**(1. / 3.)
                block_len = max(block_len, int(l_opt))
            except RuntimeError:
                print("Error - curve_fit failed in block_shuffle, using"
                        " block_len = %d" % (int(.05 * T)))
                # block_len = max(int(.05 * T), block_len)
        # Limit block length to a maximum of 10% of T
        block_len = min(block_len, int(0.1 * T))
        return block_len


    def _get_shuffle_dist( self, X,Y,Z,
                            sig_samples, 
                            sig_blocklength=None,
                            verbosity=0):
        """Returns shuffle distribution of test statistic.

        The rows in array corresponding to the X-variable are shuffled using
        a block-shuffle approach.

        Parameters
        ----------
        array : array-like
            input unshuffled array

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        sig_samples : int, optional (default: 100)
            Number of samples for shuffle significance test.

        sig_blocklength : int, optional (default: None)
            Block length for block-shuffle significance test. If None, the
            block length is determined from the decay of the autocovariance as
            explained in [1]_.

        verbosity : int, optional (default: 0)
            Level of verbosity.

        Returns
        -------
        null_dist : array of shape (sig_samples,)
            Contains the sorted test statistic values estimated from the
            shuffled arrays.
        """
        array = self.array
        N, dim, T = array.shape

        # x_indices = list(list(zip(*X))[0]); y_indices = list(list(zip(*Y))[0])
        x_indices = list(list(zip(*Y))[0])
        if len(Z)> 0:
            z_indices = list(list(zip(*Z))[0])
        else:
            z_indices = Z

        array_shuffled = np.copy(array)
        null_dist = np.zeros(sig_samples)
        dim_x = len(x_indices)
        
        
        for sam in range(sig_samples):

            for n in range(N):
                if sig_blocklength is None:
                    sig_blocklength = self._get_block_length(array[n], X,Y,Z,
                                                                mode='significance')

                n_blks = int(math.floor(float(T)/sig_blocklength))
                # print 'n_blks ', n_blks
                # if verbosity > 2:
                #     print("            Significance test with block-length = %d "
                #             "..." % (sig_blocklength))

                
                block_starts = np.arange(0, T - sig_blocklength + 1, sig_blocklength)
                # print(block_starts, block_starts.shape)
                # # Dividing the array up into n_blks of length sig_blocklength may
                # # leave a tail. This tail is later randomly inserted
                tail = array[n, x_indices, n_blks*sig_blocklength:]
            # print(sam)
                blk_starts = self.random_state.permutation(block_starts)[:n_blks]

                x_shuffled = np.zeros(( dim_x, n_blks*sig_blocklength),
                                    dtype=array[n].dtype)
                # print(x_shuffled.shape)
                # x_shuffled_new = np.zeros(( dim_x, T),
                #                         dtype=array.dtype)
                for i, index in enumerate(x_indices):
                    for blk in range(sig_blocklength):  
                                        
                        x_shuffled[i, blk::sig_blocklength] = \
                                array[n, index, blk_starts + blk]

                # Insert tail randomly somewhere
                if tail.shape[1] > 0:
                    insert_tail_at = self.random_state.choice(block_starts)
                    x_shuffled = np.insert(x_shuffled, insert_tail_at,
                                    tail.T, axis=1)

                for i, index in enumerate(x_indices):
                    array_shuffled[n,index] = x_shuffled[i]
            # else:
            #     rng = np.random.default_rng(seed=None)
            #     order = rng.permutation(T).astype('int32')
            #     array_shuffled[:, x_indices[0]] = array_shuffled[:, x_indices[0], order]

            print('shuffle v',self.independence_measure(array_shuffled,X=X,Y=Y,Z=Z))
            null_dist[sam] = self.independence_measure(array_shuffled,X=X,Y=Y,Z=Z)
            # print(sam,null_dist[sam])

        return null_dist
    
    def get_shuffle_significance(self, X, Y, Z, value,
                                 return_null_dist=False):
        """Returns p-value for nearest-neighbor shuffle significance test.

        For non-empty Z, overwrites get_shuffle_significance from the parent
        class  which is a block shuffle test, which does not preserve
        dependencies of X and Y with Z. Here the parameter shuffle_neighbors is
        used to permute only those values :math:`x_i` and :math:`x_j` for which
        :math:`z_j` is among the nearest niehgbors of :math:`z_i`. If Z is
        empty, the block-shuffle test is used.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        value : number
            Value of test statistic for unshuffled estimate.

        Returns
        -------
        pval : float
            p-value
        """
        array = self.array
        N, dim, T = array.shape

        # Skip shuffle test if value is above threshold
        # if value > self.minimum threshold:
        #     if return_null_dist:
        #         return 0., None
        #     else:
        #         return 0.

        # max_neighbors = max(1, int(max_neighbor_ratio*T))

        x_indices = list(list(zip(*X))[0]); y_indices = list(list(zip(*Y))[0])
        if len(Z)> 0:
            z_indices = list(list(zip(*Z))[0])
        else:
            z_indices = Z
        # xyz = [x_indices,y_indices,z_indices]

        if len(z_indices) > 0 and self.shuffle_neighbors < T:
        #     if self.verbosity > 2:
        #         print("            nearest-neighbor shuffle significance "
        #               "test with n = %d and %d surrogates" % (
        #               self.shuffle_neighbors, self.sig_samples))
            
            # Get nearest neighbors around each sample point in Z
            # for n in range(N):
            #     z_array = np.fastCopyAndTranspose(array[n, z_indices, :])
            #     tree_xyz = spatial.cKDTree(z_array)
            #     neighbors[n] = tree_xyz.query(z_array,
            #                             k=self.shuffle_neighbors,
            #                             p=np.inf,
            #                             eps=0.)[1].astype(np.int32)

            null_dist = np.zeros(self.sig_samples)
            for sam in range(self.sig_samples):
                # neighbors = np.zeros((N, T, self.shuffle_neighbors))
                # Generate random order in which to go through indices loop in
                # next step
                array_shuffled = np.copy(array)
                for n in range(N):
                    z_array = np.fastCopyAndTranspose(array[n, y_indices, :])
                    # z_array = np.fastCopyAndTranspose(array[n, z_indices, :])
                    tree_xyz = spatial.cKDTree(z_array)
                    neighbors = tree_xyz.query(z_array,
                                        k=self.shuffle_neighbors,
                                        p=np.inf,
                                        eps=0.)[1].astype(np.int32)
                    # Generate random order in which to go through indices loop in
                    # next step
                    order = self.random_state.permutation(T).astype(np.int32)
                    # Shuffle neighbor indices for each sample index
                    for i in range(len(neighbors)):
                        self.random_state.shuffle(neighbors[i])
                    # neighbors = self.random_state.permuted(neighbors, axis=1)
                    
                    # Select a series of neighbor indices that contains as few as
                    # possible duplicates
                    restricted_permutation = self.get_restricted_permutation(
                            T=T,
                            shuffle_neighbors=self.shuffle_neighbors,
                            neighbors=neighbors,
                            order=order)
                    
                    
                    for i in x_indices:
                        array_shuffled[n, i] = array[n, i, restricted_permutation]
                print('shuffle value',self.independence_measure(array_shuffled,
                                                                X,Y,Z))
                if self.independence_measure(array_shuffled,X,Y,Z) > value:
                    print('large')
                null_dist[sam] = self.independence_measure(array_shuffled,
                                                                X,Y,Z)

        else:
            # null_dist = \
            #     self.parallel_shuffles_significance(X, Y ,Z, value)

            null_dist = \
                self._get_shuffle_dist( X,Y,Z,
                                        sig_samples=self.sig_samples,
                                        sig_blocklength=self.sig_blocklength,
                                        )

        pval = (null_dist >= value).mean()

        if return_null_dist:
            # Sort
            null_dist.sort()
            return pval, null_dist
        return pval
    

    def reshape_get_block_length(self, array, xyz, mode):
        """Returns optimal block length for significance and confidence tests.

        Determine block length using approach in Mader (2013) [Eq. (6)] which
        improves the method of Peifer (2005) with non-overlapping blocks In
        case of multidimensional X, the max is used. Further details in [1]_.
        Two modes are available. For mode='significance', only the indices
        corresponding to X are shuffled in array. For mode='confidence' all
        variables are jointly shuffled. If the autocorrelation curve fit fails,
        a block length of 5% of T is used. The block length is limited to a
        maximum of 10% of T.

        Mader et al., Journal of Neuroscience Methods,
        Volume 219, Issue 2, 15 October 2013, Pages 285-291

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        mode : str
            Which mode to use.

        Returns
        -------
        block_len : int
            Optimal block length.
        """
        # Inject a dependency on siganal, optimize
        from scipy import signal, optimize
        # Get the shape of the array
        dim, T = array.shape
        # Initiailize the indices
        indices = range(dim)
        if mode == 'significance':
            indices = np.where(xyz == 0)[0]

        # Maximum lag for autocov estimation
        max_lag = int(0.1*T)
        # Define the function to optimize against
        def func(x_vals, a_const, decay):
            return a_const * decay**x_vals

        # Calculate the block length
        block_len = 1
        for i in indices:
            # Get decay rate of envelope of autocorrelation functions
            # via hilbert trafo
            autocov = self._get_acf(series=array[i], max_lag=max_lag)
            autocov[0] = 1.
            hilbert = np.abs(signal.hilbert(autocov))
            # Try to fit the curve
            try:
                popt, _ = optimize.curve_fit(
                    f=func,
                    xdata=np.arange(0, max_lag+1),
                    ydata=hilbert,
                )
                phi = popt[1]
                # Formula assuming non-overlapping blocks
                l_opt = (4. * T * (phi / (1. - phi) + phi**2 / (1. - phi)**2)**2
                         / (1. + 2. * phi / (1. - phi))**2)**(1. / 3.)
                block_len = max(block_len, int(l_opt))
            except RuntimeError:
                print("Error - curve_fit failed in block_shuffle, using"
                      " block_len = %d" % (int(.05 * T)))
                # block_len = max(int(.05 * T), block_len)
        # Limit block length to a maximum of 10% of T
        block_len = min(block_len, int(0.1 * T))
        return block_len


    def reshape_get_shuffle_significance(self,  X, Y, Z, value,
                                 return_null_dist=False):
        """Returns p-value for nearest-neighbor shuffle significance test.

        For non-empty Z, overwrites get_shuffle_significance from the parent
        class  which is a block shuffle test, which does not preserve
        dependencies of X and Y with Z. Here the parameter shuffle_neighbors is
        used to permute only those values :math:`x_i` and :math:`x_j` for which
        :math:`z_j` is among the nearest niehgbors of :math:`z_i`. If Z is
        empty, the block-shuffle test is used.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        value : number
            Value of test statistic for unshuffled estimate.

        Returns
        -------
        pval : float
            p-value
        """

        array_shift, xyz = self._array_reshape(self.array, X, Y, Z)
        array_shift = array_shift.T


        dim, T = array_shift.shape

        # Skip shuffle test if value is above threshold
        # if value > self.minimum threshold:
        #     if return_null_dist:
        #         return 0., None11


        
        #     else:
        #         return 0.

        # max_neighbors = max(1, int(max_neighbor_ratio*T))
        x_indices = np.where(xyz == 0)[0]
        y_indices = np.where(xyz == 1)[0]
        z_indices = np.where(xyz == 2)[0]

        if len(z_indices) > 0 and self.shuffle_neighbors < T:
            # if self.verbosity > 2:
            #     print("            nearest-neighbor shuffle significance "
            #           "test with n = %d and %d surrogates" % (
            #           self.shuffle_neighbors, self.sig_samples))

            # Get nearest neighbors around each sample point in Z
            z_array = np.fastCopyAndTranspose(array_shift[z_indices, :])
            tree_xyz = spatial.cKDTree(z_array)
            neighbors = tree_xyz.query(z_array,
                                       k=self.shuffle_neighbors,
                                       p=np.inf,
                                       eps=0.)[1].astype(np.int32)

            null_dist = np.zeros(self.sig_samples)
            for sam in range(self.sig_samples):

                # Generate random order in which to go through indices loop in
                # next step
                order = self.random_state.permutation(T).astype(np.int32)

                # Shuffle neighbor indices for each sample index
                for i in range(len(neighbors)):
                    self.random_state.shuffle(neighbors[i])
                # neighbors = self.random_state.permuted(neighbors, axis=1)
                
                # Select a series of neighbor indices that contains as few as
                # possible duplicates
                restricted_permutation = self.get_restricted_permutation(
                        T=T,
                        shuffle_neighbors=self.shuffle_neighbors,
                        neighbors=neighbors,
                        order=order)

                array_shuffled = np.copy(array_shift)
                for i in x_indices:
                    array_shuffled[i] = array_shift[i, restricted_permutation]
                print('shuffle val', self._independence_measure(array_shuffled.T,
                                                             xyz))
                null_dist[sam] = self._independence_measure(array_shuffled.T,
                                                             xyz)

        else:
            null_dist = \
                    self.reshape_get_shuffle_dist(array_shift, xyz,
                                        #    self.get_dependence_measure,
                                           sig_samples=self.sig_samples,
                                           sig_blocklength=self.sig_blocklength,
                                           )

        pval = (null_dist >= value).mean()

        if return_null_dist:
            # Sort
            null_dist.sort()
            return pval, null_dist
        return pval
    


    def reshape_get_shuffle_dist(self, array, xyz,
                          sig_samples, sig_blocklength=None,
                          ):
        """Returns shuffle distribution of test statistic.

        The rows in array corresponding to the X-variable are shuffled using
        a block-shuffle approach.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

       dependence_measure : object
           Dependence measure function must be of form
           dependence_measure(array, xyz) and return a numeric value

        sig_samples : int, optional (default: 100)
            Number of samples for shuffle significance test.

        sig_blocklength : int, optional (default: None)
            Block length for block-shuffle significance test. If None, the
            block length is determined from the decay of the autocovariance as
            explained in [1]_.

        verbosity : int, optional (default: 0)
            Level of verbosity.

        Returns
        -------
        null_dist : array of shape (sig_samples,)
            Contains the sorted test statistic values estimated from the
            shuffled arrays.
        """

        dim, T = array.shape

        x_indices = np.where(xyz == 0)[0]
        dim_x = len(x_indices)

        if sig_blocklength is None:
            sig_blocklength = self.reshape_get_block_length(array, xyz,
                                                     mode='significance')

        n_blks = int(math.floor(float(T)/sig_blocklength))
        # print 'n_blks ', n_blks
        # if verbosity > 2:
        #     print("            Significance test with block-length = %d "
        #           "..." % (sig_blocklength))

        array_shuffled = np.copy(array)
        block_starts = np.arange(0, T - sig_blocklength + 1, sig_blocklength)

        # Dividing the array up into n_blks of length sig_blocklength may
        # leave a tail. This tail is later randomly inserted
        tail = array[x_indices, n_blks*sig_blocklength:]

        null_dist = np.zeros(sig_samples)
        for sam in range(sig_samples):

            blk_starts = self.random_state.permutation(block_starts)[:n_blks]

            x_shuffled = np.zeros((dim_x, n_blks*sig_blocklength),
                                  dtype=array.dtype)

            for i, index in enumerate(x_indices):
                for blk in range(sig_blocklength):
                    x_shuffled[i, blk::sig_blocklength] = \
                            array[index, blk_starts + blk]

            # Insert tail randomly somewhere
            if tail.shape[1] > 0:
                insert_tail_at = self.random_state.choice(block_starts)
                x_shuffled = np.insert(x_shuffled, insert_tail_at,
                                       tail.T, axis=1)

            for i, index in enumerate(x_indices):
                array_shuffled[index] = x_shuffled[i]

            print('shuffle v', self._independence_measure(array_shuffled.T,
                                                             xyz))

            null_dist[sam] = self._independence_measure(array=array_shuffled.T,
                                                xyz=xyz)

        return null_dist
    

    def _get_single_residuals(self, array, xyz, target_var,
                              standardize=True,
                              return_means=False):
        """Returns residuals of linear multiple regression.

        Performs a OLS regression of the variable indexed by target_var on the
        conditions Z. Here array is assumed to contain X and Y as the first two
        rows with the remaining rows (if present) containing the conditions Z.
        Optionally returns the estimated regression line.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        target_var : {0, 1}
            Variable to regress out conditions from.

        standardize : bool, optional (default: True)
            Whether to standardize the array beforehand. Must be used for
            partial correlation.

        return_means : bool, optional (default: False)
            Whether to return the estimated regression line.

        Returns
        -------
        resid [, mean] : array-like
            The residual of the regression and optionally the estimated line.
        """

        dim, T = array.shape
        dim_z = (xyz == 2).sum()

        # Standardize
        if standardize:
            array -= array.mean(axis=1).reshape(dim, 1)
            std = array.std(axis=1)
            for i in range(dim):
                if std[i] != 0.:
                    array[i] /= std[i]
            if np.any(std == 0.) and self.verbosity > 0:
                warnings.warn("Possibly constant array!")
            # array /= array.std(axis=1).reshape(dim, 1)
            # if np.isnan(array).sum() != 0:
            #     raise ValueError("nans after standardizing, "
            #                      "possibly constant array!")

        y = np.fastCopyAndTranspose(array[np.where(xyz==target_var)[0], :])

        if dim_z > 0:
            z = np.fastCopyAndTranspose(array[np.where(xyz==2)[0], :])
            beta_hat = np.linalg.lstsq(z, y, rcond=None)[0]
            mean = np.dot(z, beta_hat)
            resid = y - mean
        else:
            resid = y
            mean = None

        if return_means:
            return (np.fastCopyAndTranspose(resid), np.fastCopyAndTranspose(mean))

        return np.fastCopyAndTranspose(resid)

    def get_partial_correlation(self, array, X, Y, Z):
        """Return multivariate kernel correlation coefficient.

        Estimated as some dependency measure on the
        residuals of a linear OLS regression.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        Returns
        -------
        val : float
            Partial correlation coefficient.
        """

        array_shift, xyz = self._array_reshape(array, X, Y, Z)
        array = array_shift.T
        dim, T = array.shape
        dim_x = (xyz==0).sum()
        dim_y = (xyz==1).sum()

        x_vals = self._get_single_residuals(array, xyz, target_var=0)
        y_vals = self._get_single_residuals(array, xyz, target_var=1)

        array_resid = np.vstack((x_vals.reshape(dim_x, T), y_vals.reshape(dim_y, T)))
        xyz_resid = np.array([index_code for index_code in xyz if index_code != 2])

        val = self.mult_corr(array_resid, xyz_resid)
        return val

    def mult_corr(self, array, xyz, standardize=True):
        """Return multivariate dependency measure.

        Parameters
        ----------
        array : array-like
            data array with X, Y in rows and observations in columns

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        standardize : bool, optional (default: True)
            Whether to standardize the array beforehand. Must be used for
            partial correlation.

        Returns
        -------
        val : float
            Multivariate dependency measure.
        """

        dim, n = array.shape
        dim_x = (xyz==0).sum()
        dim_y = (xyz==1).sum()
        # print("XYZ.T", array.shape)

        # Standardize
        if standardize:
            array -= array.mean(axis=1).reshape(dim, 1)
            std = array.std(axis=1)
            for i in range(dim):
                if std[i] != 0.:
                    array[i] /= std[i]
            # if np.any(std == 0.) and self.verbosity > 0:
            #     warnings.warn("Possibly constant array!")
            # array /= array.std(axis=1).reshape(dim, 1)
            # if np.isnan(array).sum() != 0:
            #     raise ValueError("nans after standardizing, "
            #                      "possibly constant array!")

        x = array[np.where(xyz==0)[0]]
        y = array[np.where(xyz==1)[0]]

        # if self.correlation_type == 'max_corr':
            # Get (positive or negative) absolute maximum correlation value
        corr = np.corrcoef(x, y)[:len(x), len(x):].flatten()
        val = corr[np.argmax(np.abs(corr))]

            # val = 0.
            # for x_vals in x:
            #     for y_vals in y:
            #         val_here, _ = stats.pearsonr(x_vals, y_vals)
            #         val = max(val, np.abs(val_here))
        
        # elif self.correlation_type == 'linear_hsci':
        #     # For linear kernel and standardized data (centered and divided by std)
        #     # biased V -statistic of HSIC reduces to sum of squared inner products
        #     # over all dimensions
        #     val = ((x.dot(y.T)/float(n))**2).sum()
        # else:
        #     raise NotImplementedError("Currently only"
        #                               "correlation_type == 'max_corr' implemented.")

        return val
    
    def get_analytic_significance(self, array, X, Y, Z):
        """Returns analytic p-value depending on correlation_type.

        Assumes two-sided correlation. If the degrees of freedom are less than
        1, numpy.nan is returned.

        Parameters
        ----------
        value : float
            Test statistic value.

        T : int
            Sample length

        dim : int
            Dimensionality, ie, number of features.

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        Returns
        -------
        pval : float or numpy.nan
            P-value.
        """
        # Get the number of degrees of freedom
        value = self.get_partial_correlation(array, X, Y, Z)

        array_shift, xyz = self._array_reshape(array, X, Y, Z)
        T, dim = array_shift.shape
        deg_f = T - dim

        dim_x = (xyz==0).sum()
        dim_y = (xyz==1).sum()

        # if self.correlation_type == 'max_corr':
        if deg_f < 1:
            pval = np.nan
        elif abs(abs(value) - 1.0) <= sys.float_info.min:
            pval = 0.0
        else:
            trafo_val = value * np.sqrt(deg_f/(1. - value*value))
            # Two sided significance level
            pval = stats.t.sf(np.abs(trafo_val), deg_f) * 2
        # else:
        #     raise NotImplementedError("Currently only"
        #                               "correlation_type == 'max_corr' implemented.")

        # Adjust p-value for dimensions of x and y (conservative Bonferroni-correction)
        pval *= dim_x*dim_y

        return value, pval

