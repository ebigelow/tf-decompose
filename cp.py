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

from tqdm import tqdm, trange


import logging
logging.basicConfig(filename='loss.log', level=logging.DEBUG)
_log = logging.getLogger('CP')


def frobenius(X):
    return tf.reduce_sum(X ** 2) ** 0.5

def shuffled(ls):
    return sorted(list(ls), key=lambda _: np.random.rand())


def bilinear(A, B):
    """
    TODO: describe
    * handles if i,j are lists of indices (i.e. A/B are tensors)

    C_ijk = A_ik       B_jk
          = A'_ijk  *  B'_ijk

    Table of tensor shapes
    ------------------------------------------------------------
      A   .   B          return
    (i,k) . (j,k)    =>  (i,j,k)
    ------------------------------------------------------------
    (i,k)            =>  (i,1,k)        # expand dims
    (j,k)            =>  (1,j,k)
    (i,1,k)          =>  (i,j,k)        # tile expanded
    (1,j,k)          =>  (i,j,k)
    (i,j,k) .* (i,j,k)                  # elem-wise multiply
    ------------------------------------------------------------
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


        Table of tensor shapes:
        ---------------------------------------------------------------------------------
        (i,r) . (j,r) . (k,r)    =>  (i,j,k)       # goal: reconstruct `X` from `self.U`
        ---------------------------------------------------------------------------------
        (j,r) . (k,r)            =>  (j,k,r)       # bilinear interpolation
        (j,k,r)                  =>  (j*k,r)       # reshape
        (i,r) . (r,j*k)          =>  (i,j*k)       # combine components
        (i,j*k)                  =>  (i,j,k)       # reshape
        ---------------------------------------------------------------------------------
        """
        with tf.name_scope('X_predict'):
            # import ipdb; ipdb.set_trace()

            reduced = reduce(bilinear, self.U[1:][::-1])
            rs = reduced.shape.as_list()
            reshaped = tf.reshape(reduced, [np.prod(rs[:-1]), rs[-1]])

            with tf.name_scope('interpolate'):
                interpolated = tf.matmul(self.U[0], tf.transpose(reshaped))

            self.X_predict = tf.reshape(interpolated, self.shape, name='reshape')

    def get_train_ops(self, X_var, minimizer):
        """
        get optimizers   Optimizer(..., vars=[U_i])    for each i

        """
        errors = X_var - self.X_predict
        loss = frobenius(errors)
        return [(minimizer(loss, n), loss) for n in range(self.order)]

    def get_train_op(self, X_var, minimizer, n):
        """
        get optimizers   Optimizer(..., vars=[U_i])    for each i

        """
        errors = X_var - self.X_predict
        loss = frobenius(errors)
        return [(minimizer(loss, n), loss) for n in range(self.order)]

    def cp_als(self, X, optimizer, epochs=10):
        """
        Example:
          from scipy.io.matlab import loadmat
          X = np.load('data_tensor.npy')
          M = KruskalTensor(X.shape, rank=50)
          M.cp_als(X, tf.train.AdagradOptimzer(0.01), epochs=10)

        """
        # get_vars = lambda n: tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.U[n])
        minimizer = lambda cost, n: optimizer.minimize(cost, var_list=[self.U[n]])

        # import ipdb; ipdb.set_trace()
        X_var = tf.Variable(X)
        train_ops = self.get_train_ops(X_var, minimizer)

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)

            for e in trange(epochs):
                for alt, (train_op, loss) in enumerate(shuffled(train_ops)):
                    _, loss = sess.run([train_op, loss], feed_dict={X_var: X})
                    _log.debug('[%3d:%3d] loss: %.5f' % (e, alt, loss))

            print 'final loss: %.5f' % loss
            return sess.run(self.X_predict)




# TODO remember that U_1 is actually U_1 * Lambda   -- will this cause any issues?






# # ========================================================================================================
#
# def unfold(X, shape, n):
#     """
#     Unfold a TF tensor of shape (d_1, d_2, ..., d_N)
#         into a matrix   (d_n, D' )
#         where    D' = d_1 * d_2 * ... * d_n-1 * d_n+1 * ... * d_N
#     """
#     s = shape[n]
#     d1 = tf.prod(shape[:n])
#     d2 = tf.prod(shape[(n+1):])
#
#     X_ = tf.reshape(X, [d1, s, d2])
#     return tf.reshape(X_, [s, d1*d2])   # TODO this and the line above it are likely to break
#
# def init_nvecs(X, rank, do_flipsign=True):
#     """
#     Eigendecomposition of mode-n unfolding of a tensor
#
#     TODO
#     - will there be issues if we don't handle sparsity specially?
#     """
#     Y = tf.matmul(X, X.T)
#
#     N = Y.shape[0]
#     _, U = eigh(Y, )
#     # reverse order of eigenvectors such that eigenvalues are decreasing
#     U = array(U[:, ::-1])
#     # flip sign
#     if do_flipsign:
#         U = flipsign(U)
#
#     n_eigvals = (N - rank, N - 1)
#
#     e,v = tf.self.adjoint_eig(X)
#     return e

def outer(U, V):
    """
    outer tensor product of two arbitrarily dimension tensors
    (i,j,k) . (l,m)  -->  (i,j,k,l,m)
    """
    U_ = tf.expand_dims(U, -1)
    V_ = tf.expand_dims(V,  0)
    return tf.matmul(U_, V_)
