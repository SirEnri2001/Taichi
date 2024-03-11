import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.decomposition import PCA
X = np.random.rand(10000,10000)
X = X + X.T
# cov = np.cov(X.T)
eig_val, eig_vec = np.linalg.eigh(X)
print('Eigenvalues = ', eig_val)
print('Eigenvectors = ', eig_vec)