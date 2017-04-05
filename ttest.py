import numpy as np
import tensorflow as tf
from scipy.io.matlab import loadmat
from ttensor import TuckerTensor

mat = loadmat('data/bread/brod.mat')
X = mat['X'].reshape([10,11,8])

T = TuckerTensor(X.shape, ranks=[3,3,3], regularize=0.0, init='random')

#X_predict = T.train_als(X, tf.train.AdadeltaOptimizer(0.05), epochs=15000)
X_predict = T.hooi(X, epochs=100)
#X_predict = T.hosvd(X)

np.save('X_tucker.npy', X_predict)
