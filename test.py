import numpy as np
from scipy.io.matlab import loadmat
import tensorflow as tf
from ktensor import KruskalTensor

mat = loadmat('data/bread/brod.mat')
X = mat['X'].reshape([10,11,8])

T = KruskalTensor(X.shape, rank=3, regularize=1e-6, init='nvecs', X_data=X)
# T = KruskalTensor(X.shape, rank=3, regularize=1e-6)
X_predict = T.train_als(X, tf.train.GradientDescentOptimizer(0.005), epochs=15000)

np.save('X_predict.npy', X_predict)
