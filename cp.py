"""

TF implementation of CP decomposition, using least-squares


map  X  ->  A . B . C ...


general problem:

    min_X* ||X - X*||     (X is tensor of order d)


ALS formulation:

    min_A* || X  -  A* (C . B)^T ||_F



with:
    X* = sum_{r=1}^R \lambda_r  a_r  .  b_r  .  c_r

    A* = A . diag(\lambda)




X           (M x N x P)

A           (M x R)
B           (N x R)
C           (P x R)

C  0  B     (P x N x R)





general form:

X*  =  A* (C  .  B)^T

X*  = U_1* ( U_N  .  U_{N-1}  .  U_{N-2} ...  .  U_2 ) ^T


"""

import numpy as np
import tensorflow as tf
from tqdm import trange

import logging
logging.basicConfig(filename='loss.log', level=logging.DEBUG)
_log = logging.getLogger('CP')


def shuffled(ls):
    return sorted(list(ls), key=lambda _: np.random.rand())

def frobenius(X):
    return tf.reduce_sum(X ** 2) ** 0.5

def bilinear(A, B):
    """
    Return the bilinear tensor product of two tensors A,B.

    Note that A,B must share their final dimension, all other
      dimensions are used for the bilinear product.

    C_ijk = A_ik    *  B_jk      (i,j,k index of C is equal to A_ik times B_jk )
          = A'_ijk  *  B'_ijk    (tiling A,B along along respective first axis)

    Tensor shapes (where i and j may refer to lists of indices):
          A      B           C
        (i,k)  (j,k)  =>  (i,j,k)
    """
    a_shape, a_order = (A.get_shape().as_list(), len(A.get_shape()))
    b_shape, b_order = (B.get_shape().as_list(), len(B.get_shape()))

    # Expand dimensions
    A_expand = tf.expand_dims(A, -2)
    for _ in range(b_order - 2):
        A_expand = tf.expand_dims(A_expand, -2)

    B_expand = tf.expand_dims(B, 0)
    for _ in range(a_order - 2):
        B_expand = tf.expand_dims(B_expand, 0)

    # Tile expanded tensors
    a_shape_new = [1] * (a_order + b_order - 1)
    a_shape_new[a_order-1:-1] = b_shape[:-1]
    A_tiled = tf.tile(A_expand, a_shape_new)

    b_shape_new = [1] * (b_order + a_order - 1)
    b_shape_new[0:a_order-1] = a_shape[:-1]
    B_tiled = tf.tile(B_expand, b_shape_new)

    # Element-wise product of tiled tensors
    return A_tiled * B_tiled


class KruskalTensor:
    """
    Used for CP decomposition of a tensor.

    There will be a little messiness at some points because tf.einsum doesn't allow
      sloppy formatting, e.g. `tf.einsum('i...', '...i')`

    """
    def __init__(self, shape, rank, init='random', dtype=tf.float64):
        self.shape = shape
        self.order = len(shape)
        self.rank  = rank
        self.dtype = dtype
        self.init_random()
        self.init_reconstruct()

    def init_random(self, a=0.0, b=1.0):
        """
        Init component matrices `U` with random vals in the interval [a,b).

        """
        self.Lambda = tf.Variable(tf.random_uniform(
            (self.rank, self.rank), a, b, self.dtype), name='Lambda')

        with tf.name_scope('U'):
            self.U = [None] * self.order
            for n in range(0, self.order):
                shape = (self.shape[n], self.rank)
                self.U[n] = tf.Variable(tf.random_uniform(
                    shape, minval=a, maxval=b, dtype=self.dtype), name=str(n))

    def init_reconstruct(self):
        """
        Reconstruct predicted data tensor `X` with components `self.U`.

        We first compute a bilinear interpolation (Khatri-Rao column-wise tensor
          product), then reshape the output. Note that we use `reduce` so we
          don't have to tile `self.order` full component tensors, each with
          `self.shape` dimensionality.

        """
        with tf.name_scope('X_predict'):
            reduced = reduce(bilinear, self.U[1:][::-1])
            rs = reduced.shape.as_list()
            reshaped = tf.reshape(reduced, [np.prod(rs[:-1]), rs[-1]])

            u0 = tf.matmul(self.U[0], self.Lambda)
            with tf.name_scope('interpolate'):
                interpolated = tf.matmul(u0, tf.transpose(reshaped))

            self.X_predict = tf.reshape(interpolated, self.shape, name='reshape')

    def get_train_ops(self, X_var, optimizer):
        """
        Get separate optimizers for each component matrix.

        """
        errors = X_var - self.X_predict
        loss_op = frobenius(errors)

        min_U = [ optimizer.minimize(loss_op, var_list=[self.U[n]])
                    for n in range(self.order) ]
        min_Lambda = optimizer.minimize(loss_op, var_list=[self.Lambda])
        train_ops = min_U + [min_Lambda]

        return loss_op, train_ops

    def cp_als(self, X, optimizer, epochs=10):
        """
        Use alt. least-squares to find the CP decomposition of tensor `X`.

        """
        X_var = tf.Variable(X)
        loss_op, train_ops = self.get_train_ops(X_var, optimizer)

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)

            for e in trange(epochs):
                for alt, train_op in enumerate(shuffled(train_ops)):
                    _, loss = sess.run([train_op, loss_op], feed_dict={X_var: X})
                    _log.debug('[%3d:%3d] loss: %.5f' % (e, alt, loss))

            print 'final loss: %.5f' % loss
            return sess.run(self.X_predict)
