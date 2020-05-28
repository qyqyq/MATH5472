import time
import numpy as np
from copy import deepcopy as dcpy
from sklearn.datasets import make_low_rank_matrix as mlrm
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pylab as plt

# quick print and check the shape of matrix to debug
def prt(str, X):
    print(str, X.shape)

class softImpute_ALS:
    def __init__(self, X_truth, X, _lambda, rank, nan_zero, maxiter = 600): # nan:1, zero:0
        # Parameter
        self.X_truth = X_truth
        self.X = X
        self.m, self.n = self.X.shape
        self._lambda = _lambda
        self.r = rank
        self.maxiter = maxiter
        self.eps = 1e-7
        
        # Initialization
        # since nan*0 = nan, we need to make an auxiliary matrix for projection
        self.X_fill = self.X.copy()
        if nan_zero:
            self.omega = (np.isnan(X)==False).astype(int)
            col_mean = np.nanmean(self.X_fill, axis = 0)
        else:
            self.omega = np.array(np.nonzero(X))
            col_mean = np.mean(self.X_fill, axis = 0)
        
        col_mean = np.nanmean(self.X_fill, axis = 0)
        np.copyto(self.X_fill, col_mean, where=np.isnan(self.X_fill))
        self.P_X = self.X_fill * self.omega
        self.P_XT = self.P_X.T
        
        self.u = np.random.rand(self.m, self.r)
        self.d = np.identity(self.r)
        self.v = np.random.rand(self.n, self.r)
        
        self.time_seq = []
        self.mse_seq = []
        
    def softimpute_als(self):
        X = dcpy(self.X_fill)
        U = dcpy(self.u)
        D = dcpy(self.d)
        V = dcpy(self.v)
        # for calculating Frobenius norm to check convergence
        U_old = dcpy(self.u)
        D_old = dcpy(self.d)
        V_old = dcpy(self.v)
        it = 0
        start_time = time.time()
        while it < self.maxiter:
            it = it + 1
            # step 1
            A = U @ D
            B = V @ D
            # step 2
            # 2(a)
            ABT = A @ B.T
            P_ABT = ABT * self.omega
            X_star = self.P_X - P_ABT + ABT
            # prt('X_star:',X_star)
            # 2(b)
            #prt('D', D)
            D2LI_inv = np.linalg.inv(D * D + self._lambda * np.identity(self.r))
            B_tildeT = D2LI_inv @ D @ U.T @ X_star
            B_tilde = B_tildeT.T
            #2(C)
            # if there exists nan in matrix, SVD will not converge
            V, D2, _ = np.linalg.svd( B_tilde @ D, full_matrices = False )
            D = np.diag(np.sqrt(D2))
            B = V @ D
            
            #step 3
            BAT = B @ A.T
            P_BAT = BAT * self.omega.T
            X_starT = self.P_XT - P_BAT + BAT
            D2LI_inv = np.linalg.inv(D * D + self._lambda * np.identity(self.r))
            A_tildeT = D2LI_inv @ D @ V.T @ X_starT
            A_tilde = A_tildeT.T
            U, D2, _ = np.linalg.svd( A_tilde @ D, full_matrices = False )
            D = np.diag(np.sqrt(D2))
            A = U @ D
            
            cross_term = (D_old**2) @ U_old.T @ U @ (D**2) @ V.T @ V_old
            Frobenius = abs( (np.trace(D_old**4) + np.trace(D**4) - 2*np.trace(cross_term)) / np.trace(D_old**4) )
            
            if Frobenius < self.eps:
                print(Frobenius)
                break
            
            U_old = dcpy(U)
            D_old = dcpy(D)
            V_old = dcpy(V)
            
            M = X_star @ V
            u, d, RT = np.linalg.svd( M, full_matrices = False )
            d = np.fmax(d - self._lambda, 0)
            d = np.diag(d)
            v = V @ RT.T
            self.time_seq.append( time.time() - start_time )
            #self.mse_seq.append( np.sqrt(mse(X_truth, u @ d @ v.T)) )
            self.mse_seq.append( mse(X_truth, u @ d @ v.T))
            
        
        M = X_star @ V
        U, D, RT = np.linalg.svd( M, full_matrices = False )
        
        self.u = U
        #print(D)
        D = np.fmax(D - self._lambda, 0)
        self.d = np.diag(D)
        self.v = V @ RT.T
        
        end_time = time.time()
        print('number of iteration:', it)
        print('time:', end_time - start_time)