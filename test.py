import numpy as np
from scipy.io.matlab import loadmat
import tensorflow as tf
import cp

mat = loadmat('data/bread/brod.mat')
X = mat['X'].reshape([10,11,8])

T = cp.KruskalTensor(X.shape, rank=3, regularize=1e-4)
X_predict = T.cp_als(X, tf.train.GradientDescentOptimizer(0.01), epochs=3000)

np.save('X_predict.npy', X_predict)
