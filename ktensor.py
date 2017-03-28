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
from scipy.linalg import eigh
from dtensor import DecomposedTensor

from tqdm import trange
import logging
logging.basicConfig(filename='loss.log', level=logging.DEBUG)
_log = logging.getLogger('CP')


def shuffled(ls):
    return sorted(list(ls), key=lambda _: np.random.rand())

def unfold_tf(A, n):
    """
    Unfold a TF tensor of shape (d_1, d_2, ..., d_N)
        into a matrix   (d_n, D' )
        where    D' = d_1 * d_2 * ... * d_n-1 * d_n+1 * ... * d_N
    """
    shape = A.get_shape().as_list()
    idxs = [i for i,_ in enumerate(shape)]

    new_idxs = [n] + idxs[:n] + idxs[(n+1):]
    B = tf.transpose(A, new_idxs)

    dim = shape[n]
    return tf.reshape(B, [dim, -1])

def unfold_np(arr, ax):
    """
    https://gist.github.com/nirum/79d8e14da106c77c02c1
    """
    return np.rollaxis(arr, ax, 0).reshape(arr.shape[ax], -1)

def nvecs(X, rank, n):
    """
    TODO: describe
    """
    X_ = unfold_np(X, n)
    Y = X_.dot(X_.T)
    N = Y.shape[0]

    _, U = eigh(Y, eigvals=(N - rank, N - 1))
    # reverse order of eigenvectors such that eigenvalues are decreasing
    U = np.array(U[:, ::-1])
    return U


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



class KruskalTensor(DecomposedTensor):
    """
    Used for CP decomposition of a tensor.

    There will be a little messiness at some points because tf.einsum doesn't allow
      sloppy formatting, e.g. `tf.einsum('i...', '...i')`

    TODO
    ----
    - nvec initialization
    - compute fit as in sktensor
    - batch option for tensors too big to fit on GPU (how to do this?)

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
        Init component matrices with random vals in the interval [a,b).

        """
        self.Lambda = tf.Variable(tf.random_uniform(
            [self.rank], a, b, self.dtype), name='Lambda')

        with tf.name_scope('U'):
            self.U = [None] * self.order
            for n in range(0, self.order):

                if init == 'nvecs':
                    init_val = nvecs(X_data, self.rank, n)
                elif init == 'random':
                    shape = (self.shape[n], self.rank)
                    init_val = np.random.uniform(low=a, high=b, size=shape)

                self.U[n] = tf.Variable(init_val, name=str(n), dtype=self.dtype)

    def init_reconstruct(self):
        """
        Initialize variable for reconstructed tensor `X` with components `U`.

        We first compute a bilinear interpolation (Khatri-Rao column-wise tensor
          product), then reshape the output. Note that we use `reduce` so we
          don't have to tile `self.order` full component tensors, each with
          `self.shape` dimensionality.

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
        Efficient computation of norm for `KruskalTensor` (see Bader & Kolda).

        """
        U = tf.Variable(tf.ones((self.rank, self.rank), dtype=self.dtype))
        for n in range(self.order):
            U *= tf.matmul(tf.transpose(self.U[n]), self.U[n])

        self.norm = tf.matmul(tf.matmul(self.Lambda[None, ...], U), self.Lambda[..., None])

    def get_train_ops(self, X_var, optimizer):
        """
        Get separate optimizers for each component matrix.

        """
        errors = X_var - self.X
        loss_op = tf.reduce_sum(errors ** 2)  + (self.regularize * self.norm)

        ## This is how they do it in sktensor?
        # normX =  tf.reduce_sum(X_var ** 2)
        # normresidual = normX  +  self.norm**2 - 2 * tf.reduce_sum(self.X * X_var)
        # loss_op = (normresidual / normX)

        min_U = [ optimizer.minimize(loss_op, var_list=[self.U[n]])
                    for n in range(self.order) ]
        min_Lambda = optimizer.minimize(loss_op, var_list=[self.Lambda])
        train_ops = min_U #+ [min_Lambda]  # TODO

        return loss_op, train_ops

    def get_solution(self, X_var, n, fast_grad=True):
        """
        Faster way (may be less accurate):
          U[n]* = unfold(X,n) bilinear(U[n_ =/= n])  pinv( product( [U[n_]^T . U[n_]  for n_ =/= n] ) )

        Slower, more accurate
          U[n]* = unfold(X,n) pinv( bilinear(U[n_ =/= n]) ^T )

        Normalize columns of solution to implicitly represent Lambda
          lambda_r = ||u*_r||,  u_r = u*_r / lambda_r

        """
        X_n = unfold_tf(X_var, n)
        U_ = self.U[:n] + self.U[(n+1):]

        U_bilinear = reduce(bilinear, U_[::-1])
        U_bilinear = tf.reshape(U_bilinear, [-1, self.rank])

        if fast_grad:
            u_prod = tf.foldl(tf.multiply, [ tf.matmul(tf.transpose(u), u)  for u in U_ ])
            fast_pinv = tf.matrix_inverse(u_prod)
            solution = tf.matmul(tf.matmul(X_n, U_bilinear), fast_pinv)
        else:
            slow_pinv = tf.matrix_inverse( tf.transpose(U_bilinear) )
            solution = tf.matmul(X_n, slow_pinv)

        # Normalize columns of solution
        lambda_r = tf.reduce_sum(solution ** 2, axis=0) ** 0.5
        solution = solution / lambda_r

        update_op = self.U[n].assign(solution)
        return update_op

    def als_solve(self, X_data, epochs=10, fast_grad=True):
        """
        Use alt. least-squares to find the Tucker decomposition of tensor `X`.

        """
        X_var = tf.Variable(X_data)
        loss_op = tf.reduce_sum((X_var - self.X) ** 2)
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)

            for e in trange(epochs):
                for alt, n in enumerate(shuffled(range(self.order))):

                    update_op = self.get_solution(X_var, n, fast_grad=fast_grad)

                    _, loss = sess.run([update_op, loss_op], feed_dict={X_var: X_data})
                    _log.debug('[%3d:%3d] loss: %.5f' % (e, alt, loss))

            print 'final loss: %.5f' % loss
            return sess.run(self.X)
