import numpy as np
import matplotlib.pylab as plt
from copy import deepcopy as dcpy
from sklearn.metrics import mean_squared_error as mse
from sklearn.datasets import make_low_rank_matrix as mlrm

import ALS
import softImpute
import softImpute_ALS

# Simulation Data:
m = 100
n = 50
X_truth = 100 * mlrm(n_samples = m, n_features = n, effective_rank = 10)
print('ground truth:', X_truth)

X_missing = X_truth.copy()
missing_rate = 0.9
num_nan = int(missing_rate * m * n) # number of nan
X_missing.ravel()[np.random.choice(X_missing.size, num_nan, replace=False)] = np.nan

als = ALS.alterLeastSquares(X_truth, X_missing, _lambda = 10, rank = 30)
als.alterleastsquares()
print('MSE of ALS:', mse(X_truth, als.A @ als.B.T))

SI = softImpute.softImpute(X_truth, X_missing, _lambda = 5, nan_zero = 1)
SI.softimpute()
print('MSE of softImpute:', mse(X_truth, SI.M))

SI_ALS = softImpute_ALS.softImpute_ALS(X_truth, X_missing, _lambda = 20, rank = 30, nan_zero = 1)
SI_ALS.softimpute_als()
X_completion = SI_ALS.u @ SI_ALS.d @ SI_ALS.v.T
print('MSE of softImpute-ALS:', mse(X_truth, X_completion))