import time
import numpy as np
from copy import deepcopy as dcpy
from sklearn.datasets import make_low_rank_matrix as mlrm
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pylab as plt

# quick print and check the shape of matrix to debug
def prt(str, X):
    print(str, X.shape)

class softImpute:
    def __init__(self, X_truth, X, _lambda, nan_zero, maxiter = 300):
        self.X_truth = X_truth
        self.X = X
        self.maxiter = maxiter
        self.m, self.n = self.X.shape
        self._lambda = _lambda
        self.eps = 1e-8
        
        self.M = np.random.rand(self.m, self.n)
        
        if nan_zero:
            self.omega = (np.isnan(X)==False).astype(int)
        else:
            self.omega = np.array(np.nonzero(X))
        # since nan*0 = nan, we need to make an auxiliary matrix for projection
        self.X_fill = self.X.copy()
        col_mean = np.nanmean(self.X_fill, axis = 0)
        np.copyto(self.X_fill, col_mean, where=np.isnan(self.X_fill))
        self.P_X = self.X_fill * self.omega
        
        self.time_seq = []
        self.mse_seq = []
    
    def softimpute(self):
        it = 0
        X_hat = dcpy(self.X)
        M = dcpy(self.M)
        start_time = time.time()
        while it < self.maxiter:
            it = it + 1
            X_hat = self.P_X - M * self.omega + M
            U, D, VT = np.linalg.svd(X_hat, full_matrices = False)
            #prt('U', U)
            #prt('D', D)
            #prt('V.T', VT)
            SD = np.diag(np.fmax(D - self._lambda, 0))
            M = U @ SD @ VT
            
            diff = np.linalg.norm(M - self.M, ord = 'fro')
            if diff < self.eps:
                break
            
            self.M = dcpy(M)
            
            self.time_seq.append( time.time()-start_time )
            #self.mse_seq.append( np.sqrt(mse(self.X_truth, M)) )
            self.mse_seq.append( mse(self.X_truth, M))
        
        end_time = time.time()
        print('number of iteration:', it)
        print('time:', end_time - start_time)