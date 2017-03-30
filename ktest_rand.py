import numpy as np
from scipy.io.matlab import loadmat
import tensorflow as tf
from ktensor import KruskalTensor

X = np.load('./data/rand.npy')

T = KruskalTensor(X.shape, rank=200, regularize=1e-5, init='nvecs', X_data=X)
X_predict = T.train_als(X, tf.train.AdadeltaOptimizer(0.05), epochs=300)
# X_predict = T.train_als(X, tf.train.GradientDescentOptimizer(0.001), epochs=1000)


np.save('rand_predict.npy', X_predict)
