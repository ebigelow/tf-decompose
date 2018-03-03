
import numpy as np
import tensorflow as tf
from dtensor import DecomposedTensor
from utils import nvecs, bilinear
from functools import reduce

class KruskalTensor(DecomposedTensor):
    """
    Used to represent CP decomposition of a tensor.

    Arguments
    ---------
    shape (list) : list of ints representing the shape of the input tensor
    rank (int) : reduced rank of decomposed tensor
    regularize (float) : parameter for L_2 norm on reconstructed tensor
    dtype (tf.dtype) : dtype of component tensors
    init (string) : component inialization method ('random' | 'nvecs')
    X_data (np.ndarray) : input tensor, only necessary if  init='nvecs'

    """
    def __init__(self, shape, rank, regularize=1e-5,
                 dtype=tf.float64, init='random', X_data=None):
        self.shape = shape
        self.order = len(shape)
        self.rank  = rank
        self.regularize = regularize
        self.dtype = dtype
        self.init_components(init, X_data)
        self.init_reconstruct()
        self.init_norm()

    def init_components(self, init, X_data, a=0.0, b=1.0):
        """
        Initialize component matrices with nvecs or random samples from [a,b).

        """
        self.Lambda = tf.Variable(tf.random_uniform(
            [self.rank], a, b, self.dtype), name='Lambda')

        with tf.name_scope('U'):
            self.U = [None] * self.order

            for n in range(self.order):
                if init == 'nvecs':
                    init_val = nvecs(X_data, self.rank, n)
                elif init == 'random':
                    shape = (self.shape[n], self.rank)
                    init_val = np.random.uniform(low=a, high=b, size=shape)
                self.U[n] = tf.Variable(init_val, name=str(n), dtype=self.dtype)

    def init_reconstruct(self):
        """
        Initialize variable for reconstructed tensor `X` with components `U`.

        Compute a bilinear interpolation (Khatri-Rao column-wise tensor
          product) folded (reduced) over component matrices U, then reshape.

        """
        with tf.name_scope('X'):
            reduced = reduce(bilinear, self.U[1:][::-1])
            rs = reduced.shape.as_list()
            reduced = tf.reshape(reduced, [np.prod(rs[:-1]), rs[-1]])

            u0 = tf.matmul(self.U[0], tf.diag(self.Lambda))
            with tf.name_scope('interpolate'):
                interpolated = tf.matmul(u0, tf.transpose(reduced))

            self.X = tf.reshape(interpolated, self.shape, name='reshape')

    def init_norm(self):
        """
        Efficient computation of L_2 norm (see Bader & Kolda, 2007).

        """
        U = tf.Variable(tf.ones((self.rank, self.rank), dtype=self.dtype))
        for n in range(self.order):
            U *= tf.matmul(tf.transpose(self.U[n]), self.U[n])

        self.norm = tf.matmul(tf.matmul(self.Lambda[None, ...], U), self.Lambda[..., None])

    def get_train_ops(self, X_var, optimizer):
        """
        Initialize separate optimizers for each component matrix.

        """
        errors = X_var - self.X
        loss_op = tf.reduce_sum(errors ** 2)  + (self.regularize * self.norm)

        min_U = [ optimizer.minimize(loss_op, var_list=[self.U[n]])
                    for n in range(self.order) ]
        min_Lambda = optimizer.minimize(loss_op, var_list=[self.Lambda])
        train_ops = min_U + [min_Lambda]

        return loss_op, train_ops
