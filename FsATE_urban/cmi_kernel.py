from __future__ import print_function
from numba import jit
import warnings
import numpy as np
from scipy.stats.contingency import crosstab
from itertools import combinations
from scipy.stats import spearmanr
from scipy import spatial, special
import math
from data_processing import DataFrame
from scipy import stats, linalg

import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.distributions.mixture_rvs import mixture_rvs
import scipy.integrate
from scipy.integrate import simps


class CMI_KERNEL():                                                                                                                                                                                                                                                              
    def __init__(self,
                 array,
                 knn = 3,
                 sig_samples=500,
                 shuffle_neighbors = 20,
                 time_lag = 1,
                 transform='standardize',
                 workers=-1):
        self.N,self.dim,self.T = array.shape
        # self.constraint = constraint
        self.array = array
        # self.n_symbs = n_symbs+1
        # self.const = np.unique(constraint,axis=0)
        self.num = self.N*self.T
        self.xyz = np.array([np.array([np.array([(n,d,tt) for tt in range(self.T)]) for d in range(self.dim)]) for n in range(self.N)])
        self.seed = 5
        self.sig_samples = sig_samples
        self.knn = knn
        self.time_lag = time_lag
        self.shuffle_neighbors = shuffle_neighbors

        self.workers = workers
        self.random_state = np.random.default_rng(self.seed)
        self.transform = transform
        


    def _array_reshape(self, array,X,Y,Z):
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

            array_shift_X = array[:,x,abs(t_min-xt):xt].ravel()
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

            array_shift_X = array[:,x,abs(t_min-xt):xt].ravel()
            array_shift_Y = array[:,y,abs(t_min-yt):].ravel()
            array_shift = np.c_[array_shift_X, array_shift_Y]
        xyz = np.array([0, 1]+list(np.ones(len(Z))*2))

        return array_shift, xyz
    
    def kde_entropy(self,array, xyz):
        """
        Parameters
        --------
        array: shape (T_after * N, dim)
        xyz: list of tuples [(x_indices, time_lag), (y_indices, time_lag), ...]

        """

        _,dim = array.shape

        x_indices = np.where(xyz==0)[0]
        y_indices = np.where(xyz==1)[0]
        z_indices = np.where(xyz==2)[0]


        # print(dim)
        if len(z_indices) == 0: # dim = 2
            kde_x = sm.nonparametric.KDEUnivariate(array[:,x_indices])
            kde_x.fit()  # Estimate the densities 
            Hx = kde_x.entropy # Hx
            kde_y = sm.nonparametric.KDEUnivariate(array[:,y_indices])
            kde_y.fit()  # Estimate the densities 
            Hy = kde_y.entropy # Hx

            size = 100
            x = np.linspace(array[:,x_indices].min(), array[:,x_indices].max(), size)
            y = np.linspace(array[:,y_indices].min(), array[:,y_indices].max(), size)
            X, Y = np.meshgrid(x,y)
            kde_xy = sm.nonparametric.KDEMultivariate(array[:,x_indices.tolist()+y_indices.tolist()], var_type='cc')
            p = kde_xy.pdf(np.c_[X.flatten(), Y.flatten()]) # using grid search as predicted data
            plogp = p*np.log(p)
            Hxy = -simps(simps(plogp.reshape(size,size),y),x)

            entropy = Hx + Hy - Hxy


        else:
            if dim < 2: # 
                kde_x = sm.nonparametric.KDEUnivariate(array[:,x_indices])
                kde_x.fit()  # Estimate the densities 
                Hx = kde_x.entropy # Hx
                kde_y = sm.nonparametric.KDEUnivariate(array[:,y_indices])
                kde_y.fit()  # Estimate the densities 
                Hy = kde_y.entropy # Hx

                size = 100
                x = np.linspace(array[:,x_indices].min(), array[:,x_indices].max(), size)
                z = np.linspace(array[:,z_indices].min(), array[:,z_indices].max(), size)
                X, Z = np.meshgrid(x,y)
                kde_xz = sm.nonparametric.KDEMultivariate(array[:,x_indices.tolist()+z_indices.tolist()], var_type='cc')
                p = kde_xz.pdf(np.c_[X.flatten(), Z.flatten()]); plogp = p*np.log(p) # using grid search as predicted data
                Hxz = -simps(simps(plogp.reshape(size,size),z),x)

                y = np.linspace(array[:,y_indices].min(), array[:,y_indices].max(), size)
                z = np.linspace(array[:,z_indices].min(), array[:,z_indices].max(), size)
                Y, Z = np.meshgrid(y,z)
                kde_yz = sm.nonparametric.KDEMultivariate(array[:,y_indices.tolist()+z_indices.tolist()], var_type='cc')
                p = kde_yz.pdf(np.c_[Y.flatten(), Z.flatten()]); plogp = p*np.log(p) # using grid search as predicted data
                Hyz = -simps(simps(plogp.reshape(size,size),z),y)




                entropy = Hxz + Hyz - Hxyz - Hz


            else:
                size = 100
                x = np.linspace(array[:,x_indices].min(), array[:,x_indices].max(), size)
                y = np.linspace(array[:,y_indices].min(), array[:,y_indices].max(), size)
                X, Y = np.meshgrid(x, y)
                # X.shape
                kde = sm.nonparametric.KDEMultivariate(array, var_type='cc')
                # kde.fit()  # Estimate the densities 
                p = kde.pdf(np.c_[X.flatten(), Y.flatten()]) # using grid search as predicted data
                plogp = p*np.log(p)
                entropy = -simps(simps(plogp.reshape(size,size),y),x)
                print(simps(simps(p.reshape(size,size),y),x))
                plt.scatter(X.ravel(),Y.ravel(),c=p)

        print(entropy)
        return entropy


    def independence_measure(self,array, X, Y, Z):

        array_shaped, xyz = self._array_reshape(array, X, Y, Z)
        TE = self.kde_entropy(array_shaped, xyz)

        return TE


        

    def parallel_shuffles(self,xyz,seed=None):
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
        
        N, dim, T = array.shape
        rng = np.random.default_rng(seed=None)
        order = rng.permutation(T).astype('int32')
        array_shuffled = np.copy(array)
        array_shuffled[:, xyz[0][0], :] = array_shuffled[:, xyz[0][0], order]

        return self.independence_measure(array_shuffled, xyz)

    def parallel_shuffles_significance(self, xyz):
        # x,y,z = X,Y,Z
        # while r < self.dim-1:
        #     for l in list(combinations(range(self.dim),2)):
        array, value = self.independence_measure(self.array, xyz)
        random_seeds = np.random.default_rng(self.seed).integers(np.iinfo(np.int32).max, size=(self.sig_samples))
        null_dist = np.zeros(self.sig_samples)

        for i, seed in enumerate(random_seeds):
            null_dist[i] = self.parallel_shuffles(xyz, seed=seed)
        
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
    
    
    
    def _time_shifted(narray, time_lag=1):
        """
        Parameters
        ----------
        narray : array of shape (N,T)

        time_lag : number of time shifts

        

        Returns
        -------
        
        """
        
        Yt = narray[:,:-time_lag]
        Yt1 = narray[:,time_lag:]
        return Yt, Yt1
    





    
    ######### select time lag ################ 

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

    def _get_block_length(self,array, xyz, mode):
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
        N, dim, T = array.shape
        # Initiailize the indices
        indices = range(dim)
        if mode == 'significance':
            indices = xyz[0]

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
            autocov = self._get_acf(series=array[:,i], max_lag=max_lag)
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
                            sig_samples, sig_blocklength=None,
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

        x_indices = list(list(zip(*X))[0]); y_indices = list(list(zip(*Y))[0])
        if len(Z)> 0:
            z_indices = list(list(zip(*Z))[0])
        else:
            z_indices = Z

        xyz = [x_indices,y_indices,z_indices]
        # x_indices, y_indices, z_indices = xyz
        # print(xyz, x_indices, y_indices, z_indices)
        dim_x = len(x_indices)

        if sig_blocklength is None:
            sig_blocklength = self._get_block_length(array, xyz,
                                                        mode='significance')

        n_blks = int(math.floor(float(T)/sig_blocklength))
        # print(sig_blocklength,n_blks)
        # print 'n_blks ', n_blks
        # if verbosity > 2:
        #     print("            Significance test with block-length = %d "
        #             "..." % (sig_blocklength))

        array_shuffled = np.copy(array)
        block_starts = np.arange(0, T - sig_blocklength + 1, sig_blocklength)
        # print(block_starts, block_starts.shape)
        # # Dividing the array up into n_blks of length sig_blocklength may
        # # leave a tail. This tail is later randomly inserted
        tail = array[:,x_indices, n_blks*sig_blocklength:]
        # print(tail.shape)
        null_dist = np.zeros(sig_samples)
        for sam in range(sig_samples):
            # print(sam)
            blk_starts = self.random_state.permutation(block_starts)[:n_blks]

            x_shuffled = np.zeros((N, dim_x, n_blks*sig_blocklength),
                                    dtype=array.dtype)
            # print(x_shuffled.shape)
            x_shuffled_new = np.zeros((N, dim_x, T),
                                    dtype=array.dtype)

            for i, index in enumerate(x_indices):
                for blk in range(sig_blocklength):
                    
                    x_shuffled[:,i, blk::sig_blocklength] = \
                            array[:,index, blk_starts + blk]

            # Insert tail randomly somewhere
            if tail.shape[2] > 0:
                
                insert_tail_at = self.random_state.choice(block_starts)
                for n in range(N):
                    x_shuffled_new[n]= np.insert(x_shuffled[n], insert_tail_at,
                                    tail[n].T, axis=1)

                for i, index in enumerate(x_indices):
                    array_shuffled[:,index] = x_shuffled_new[:,i]
            else:
                rng = np.random.default_rng(seed=None)
                order = rng.permutation(T).astype('int32')
                array_shuffled[:, x_indices[0]] = array_shuffled[:, x_indices[0], order]
            
            _,null_dist[sam] = self.independence_measure(array_shuffled,X=X,Y=Y,Z=Z)
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
        xyz = [x_indices,y_indices,z_indices]

        if len(z_indices) > 0 and self.shuffle_neighbors < T:
        #     if self.verbosity > 2:
        #         print("            nearest-neighbor shuffle significance "
        #               "test with n = %d and %d surrogates" % (
        #               self.shuffle_neighbors, self.sig_samples))
            neighbors = np.zeros((N, T, self.shuffle_neighbors))
            # Get nearest neighbors around each sample point in Z
            for n in range(N):
                z_array = np.fastCopyAndTranspose(array[n, z_indices, :])
                tree_xyz = spatial.cKDTree(z_array)
                neighbors[n] = tree_xyz.query(z_array,
                                        k=self.shuffle_neighbors,
                                        p=np.inf,
                                        eps=0.)[1].astype(np.int32)

            null_dist = np.zeros(self.sig_samples)
            for sam in range(self.sig_samples):
                # Generate random order in which to go through indices loop in
                # next step
                array_shuffled = np.copy(array)
                for n in range(N):
                    order = self.random_state.permutation(T).astype(np.int32)

                    # Shuffle neighbor indices for each sample index
                    for i in range(len(neighbors[n])):
                        self.random_state.shuffle(neighbors[n,i])
                    # neighbors = self.random_state.permuted(neighbors, axis=1)
                    
                    # Select a series of neighbor indices that contains as few as
                    # possible duplicates
                    restricted_permutation = self.get_restricted_permutation(
                            T=T,
                            shuffle_neighbors=self.shuffle_neighbors,
                            neighbors=neighbors[n],
                            order=order)
                    
                    for i in x_indices:
                        array_shuffled[n, i] = array[n, i, restricted_permutation]

                _,null_dist[sam] = self.independence_measure(array_shuffled,
                                                                X,Y,Z)

        else:
        # null_dist = \
        #         self.parallel_shuffles_significance(xyz)

            null_dist = \
                self._get_shuffle_dist( X,Y,Z,
                                        sig_samples=self.sig_samples,
                                        )

        pval = (null_dist >= value).mean()

        if return_null_dist:
            # Sort
            null_dist.sort()
            return pval, null_dist
        return pval