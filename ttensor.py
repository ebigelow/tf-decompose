

import tensorflow as tf
import numpy as np

from dtensor import DecomposedTensor
from utils import nvecs, shuffled, get_fit, refold_tf, unfold_tf

import logging
logging.basicConfig(filename='loss.log', level=logging.DEBUG)
_log = logging.getLogger('decomp')






class TuckerTensor(DecomposedTensor):
    """
    Used for Tucker decomposition of a tensor.

    """
    def __init__(self, shape, ranks, regularize=1e-5,
                 dtype=tf.float64, init='random', X_data=None):
        self.shape = shape
        self.order = len(shape)
        self.ranks = ranks if (type(ranks) is list) else [ranks]*self.order
        self.regularize = regularize
        self.dtype = dtype
        self.init_components(init, X_data)
        self.init_reconstruct()
        self.init_norm()

    def init_components(self, init, X_data, a=0.0, b=1.0):
        """
        Init component matrices with random vals in the interval [a,b).

        """
        self.G = tf.Variable(tf.random_uniform(
            self.ranks, a, b, self.dtype), name='G')

        with tf.name_scope('U'):
            self.U = [None] * self.order

            for n in range(self.order):
                if init == 'nvecs':
                    init_val = nvecs(X_data, self.ranks[n], n)
                elif init == 'random':
                    shape = (self.shape[n], self.ranks[n])
                    init_val = np.random.uniform(low=a, high=b, size=shape)
                self.U[n] = tf.Variable(init_val, name=str(n), dtype=self.dtype)

    def init_reconstruct(self):
        """
        Initialize variable for reconstructed tensor `X` with components `U`.

        """
        G_to_X = self.G
        shape = self.ranks[:]

        for n in range(self.order):
            shape[n] = self.shape[n]
            name = None if (n < self.order-1) else 'X'

            Un_mul_G = tf.matmul(self.U[n], unfold_tf(G_to_X, n))
            with tf.name_scope(name):
                G_to_X = refold_tf(Un_mul_G, shape, n)

        self.X = G_to_X

    def get_core_op(self, X_var):
        X_to_G = tf.identity(X_var)
        shape = self.ranks[:]

        for n in range(self.order):
            shape[n] = self.shape[n]

            Un_mul_X = tf.matmul(self.U[n], unfold_tf(X_to_G, n))
            X_to_G = refold_tf(Un_mul_X, shape, n)

        return tf.assign(self.G, X_to_G)

    def hosvd(self, X_data):
        """
        HOSVD

        """
        X_var = tf.Variable(X_data)
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)

            for n in shuffled(range(self.order)):
                _,u,_ = tf.svd(unfold_tf(X_var, n), 'svd%3d' % n)
                svd_op = tf.assign(self.U[n], u[:self.ranks[n]])

                # Set U[n] to the first ranks[n] left-singular values of X
                sess.run([svd_op], feed_dict={X_var: X_data})

                # Log fit after training nth component
                X_predict = sess.run(self.X)
                fit = get_fit(X_data, X_predict)
                _log.debug('[U%3d] fit: %.5f' % (n, fit))

            # Compute new core tensor value G
            core_op = self.get_core_op(X_var)
            sess.run([core_op], feed_dict={X_var: X_data})

            # Log final fit
            X_predict = sess.run(self.X)
            fit = get_fit(X_data, X_predict)
            _log.debug('[G] fit: %.5f' % fit)

            return X_predict

    def get_ortho_iter(self, X_var, n):
        Y = tf.identity(X_var)
        shape = self.ranks[:]
        idxs = [n_ for n_ in range(self.order) if n_ != n]

        for n_ in idxs:
            shape[n_] = self.shape[n_]
            name = None if (n_ < idxs[-1]) else 'Y%3d' % n_

            Un_mul_X = tf.matmul(self.U[n_], unfold_tf(Y, n_))
            with tf.name_scope(name):
                Y = refold_tf(Un_mul_X, shape, n_)

        return Y

    def hooi(self, X_data, epochs=100):
        """
        HOOI: Higher-Order Orthogonal Iteration

        """
        X_var = tf.Variable(X_data)
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            for e in range(epochs):
                for n in shuffled(range(self.order)):

                    # Get SVD for G tensor-product with all U except U[n]
                    Y = self.get_ortho_iter(X_var, n)
                    _,u,_ = tf.svd(unfold_tf(Y, n), 'svd%3d' % n)
                    svd_op = tf.assign(self.U[n], u[:self.ranks[n]])

                    # Set U[n] to the first ranks[n] left-singular values of X
                    sess.run([svd_op], feed_dict={X_var: X_data})

                    # Log fit after training nth component
                    X_predict = sess.run(self.X)
                    fit = get_fit(X_data, X_predict)
                    _log.debug('[%3d-U%3d] fit: %.5f' % (e, n, fit))

            # Compute new core tensor value G
            core_op = self.get_core_op(X_var)
            sess.run([core_op], feed_dict={X_var: X_data})

            # Log final fit
            X_predict = sess.run(self.X)
            fit = get_fit(X_data, X_predict)
            _log.debug('[G] fit: %.5f' % fit)

            return X_predict

    def get_train_ops(self, X_var, optimizer):
        """
        Get separate optimizers for each component U and core G.

        """
        errors = X_var - self.X
        loss_op = tf.reduce_sum(errors ** 2)  + (self.regularize * self.norm)

        min_U = [ optimizer.minimize(loss_op, var_list=[self.U[n]])
                    for n in range(self.order) ]
        min_G = optimizer.minimize(loss_op, var_list=[self.G])
        train_ops = min_U + [min_G]

        return loss_op, train_ops
