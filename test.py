import numpy as np
from scipy.io.matlab import loadmat
import tensorflow as tf
from ktensor import KruskalTensor

mat = loadmat('data/bread/brod.mat')
X = mat['X'].reshape([10,11,8])

T = KruskalTensor(X.shape, rank=3, regularize=1e-5, init='nvecs', X_data=X)
# X_predict = T.als_optimize(X, tf.train.AdadeltaOptimizer(0.05), epochs=15000)
# X_predict = T.train_als(X, tf.train.GradientDescentOptimizer(0.001), epochs=1000)

X_predict = T.als_solve(X, epochs=100, fast_grad=True)

np.save('X_predict.npy', X_predict)
