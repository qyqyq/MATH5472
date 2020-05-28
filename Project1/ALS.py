import time
import numpy as np
from copy import deepcopy as dcpy
from sklearn.datasets import make_low_rank_matrix as mlrm
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pylab as plt

# quick print and check the shape of matrix to debug
def prt(str, X):
    print(str, X.shape)

# Alternating Least Squares
class alterLeastSquares:
    def __init__(self, X_truth, X, _lambda, rank, maxiter = 15):
        # parameter
        self.X_truth = X_truth
        self.X = X
        self.m, self.n = self.X.shape
        self._lambda = _lambda
        self.r = rank
        self.maxiter = maxiter
        self.eps = 1e-7
        
        self.omega = (np.isnan(self.X)==False).astype(int)
        
        # initialization
        self.A = 10 * np.random.rand(self.m, self.r)
        self.B = 10 * np.random.rand(self.n, self.r)
        
        self.time_seq = []
        self.mse_seq = []
    
    def alterleastsquares(self):
        it = 0
        A = dcpy(self.A)
        B = dcpy(self.B)
        start_time = time.time()
        while it < self.maxiter:
            it = it + 1
            # Fix B and update A
            for i in range(self.m):
                BBT = np.zeros((self.r, self.r))
                XB = np.zeros((self.r))
                for j in range(self.n):
                    if self.omega[i][j] == 0:
                        continue
                    Bj = np.array([B[j, :]])
                    BBT = BBT + Bj.T @ Bj
                    XB = XB + self.X[i][j] * Bj
                A[i] = np.squeeze( np.linalg.inv(BBT + self._lambda * np.identity(self.r)) @ XB.T )
                '''
                As for the "Algorithm 5.2 Alternating least squares ALS" in paper,
                the shrinkage part might be omited,
                but actually since ALS could be considered as separate ridge regressions,
                my implementation is based on the closed form of ridge,
                adding shrinkage to the calculation of Ai and Bj.
                '''
            for j in range(self.n):
                AAT = np.zeros((self.r, self.r))
                XA = np.zeros((self.r))
                for i in range(self.m):
                    if self.omega[i][j] == 0:
                        continue
                    Ai = np.array([A[i, :]])
                    AAT = AAT + Ai.T @ Ai
                    XA = XA + self.X[i][j] * Ai
                B[j] = np.squeeze( np.linalg.inv(AAT + self._lambda * np.identity(self.r)) @ XA.T )
            '''
            It is obvious that the procedure of calculating A and B is so similar because of symmetry,
            my implementation aims to make exact alignment with the algorithm described in paper,
            but write a auxiliary function to use fixed matrix to update another would be a better implementation,
            which highly extracts same procedure, with higher code efficiency and lower risk of bugs.
            '''
            
            diff_A = np.linalg.norm(A - self.A, ord = 'fro')
            diff_B = np.linalg.norm(B - self.B, ord = 'fro')

            if diff_A < self.eps and diff_B < self.eps:
                break
            
            self.A = dcpy(A)
            self.B = dcpy(B)
            
            self.time_seq.append( time.time()-start_time )
            #self.mse_seq.append( np.sqrt(mse(self.X_truth, A @ B.T)) )
            self.mse_seq.append( mse(self.X_truth, A @ B.T))
            
        
        end_time = time.time()
        print('number of iteration:', it)
        print('time:', end_time - start_time)