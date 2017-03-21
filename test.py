from scipy.io.matlab import loadmat
import tensorflow as tf
import cp

mat = loadmat('data/bread/brod.mat')
X = mat['X'].reshape([10,11,8])

T = cp.KruskalTensor(X.shape, rank=3)
T.cp_als(X, tf.train.GradientDescentOptimizer(0.01), epochs=1)
