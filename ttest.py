import numpy as np
from scipy.io.matlab import loadmat
import tensorflow as tf
from ttensor import TuckerTensor

mat = loadmat('data/bread/brod.mat')
X = mat['X'].reshape([10,11,8])

T = TuckerTensor(X.shape, rank=3, regularize=0.0, init='nvecs', X_data=X)
X_predict = T.train_als(X, tf.train.AdadeltaOptimizer(0.05), epochs=15000)


np.save('X_tucker.npy', X_predict)
