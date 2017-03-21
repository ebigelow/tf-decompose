import logging
import numpy as np
import tensorflow as tf

_log = logging.getLogger('CP')


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



# Simple way -- let TF compute gradients!!!


def outer(U, V):
    """
    outer tensor product of two arbitrarily dimension tensors
    (i,j,k) . (l,m)  -->  (i,j,k,l,m)
    """
    U_ = tf.expand_dims(U, -1)
    V_ = tf.expand_dims(V,  0)
    return tf.matmul(U_, V_)


def bilinear_old(A, B):
    """
      A   .   B          return
    (i,k) . (j,k)    =>  (i,j,k)
    --------------------------------
    (j,k)            =>  (1,j,k)
    (1,j,k)          =>  (k,j,k)
    (i,k) . (k,j,k)  =>  (i,j,k)

    * works for i/j = {d_1, d_2, ...}
    """
    tile_shape = [1] * len(B.get_shape())
    tile_shape = A.get_shape()[-1:] + tile_shape
    B_tiled = tf.tile(tf.expand_dims(B, 0), tile_shape)
    return tf.matmul(A, B_tiled)

def bilinear(A, B):
    """
    C_ijk = A_ik      B_jk
          = A_i.k  *  B_.jk

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

    * handles if i,j are lists of indices (i.e. A/B are tensors)
    """
    ashape, alen = (A.get_shape(), len(A.get_shape()))
    bshape, blen = (B.get_shape(), len(B.get_shape()))

    # Expand dimensions
    A_expand = tf.expand_dims(A, -2)
    for _ in range(blen - 2):
        A_expand = tf.expand_dims(A_expand, -2)

    B_expand = tf.expand_dims(B, 0)
    for _ in range(alen - 2):
        B_expand = tf.expand_dims(B_expand, 0)

    # Tile expanded tensors
    ashape_new = [1] * (alen + blen - 1)
    ashape_new[alen:alen-1] = bshape[:-1]
    A_tiled = tf.tile(A_expand, ashape_new)

    bshape_new = [1] * (blen + alen - 1)
    bshape_new[0:alen-1] = ashape[:-1]
    B_tiled = tf.tile(B_expand, bshape_new)

    # Return element-wise product
    return A_tiled * B_tiled



def shuffled(ls):
    return sorted(ls, key=np.random.rand())


class KruskalTensor:
    """
    Used for CP decomposition of a tensor.

    There will be a little messiness at some points because tf.einsum doesn't allow
      sloppy formatting, e.g. `tf.einsum('i...', '...i')`

    """
    def __init__(self, shape, rank, init='random', dtype=tf.float32):
        self.shape = shape
        self.order = len(shape)
        self.rank  = rank
        self.dtype = dtype
        self.init_random()

    def init_random(self, a=0.0, b=1.0):
        """
        Init component matrices `U` with random vals in the interval [a,b).

        """
        with tf.name_scope('U'):
            self.U = [] * self.order
            for n in range(0, self.order):
                shape = (self.shape[n], self.rank)
                self.U[n] = tf.random_uniform(shape, name=str(n), minval=a, maxval=b)

    def get_train_ops(self, X_var, optimizer):
        """
        get optimizers   Optimizer(..., vars=[U_i])    for each i

        """
        X_predict = self.reconstruct()
        errors = X_var - X_predict
        loss = tf.norm.frobenius(errors)
        return [(optimizer(loss, n), loss) for n in range(self.order)]

    def reconstruct(self):
        """
        reconstruction maps components to data matrix
        (i,k) . (j,k) --> (i,j,k)

        To do this, we first compute a bilinear (khatri-rao) tensor product
        (i,k) . (j,k) --> (i*j, k)

        Then must reshape this
        (i*j, k) --> (i,j,k)

        goal     (i,l) . (j,l) . (k,l) --> (i,j,k,l)
        step 1   (i,l) . (j,l) . (k,l) --> (i*j*k, l)
        step 2   (i*j*k, l) --> (i,j,k,l)
        """
        with tf.name_scope('X_predict'):
            component_dims, shared_dims = zip(*[u.get_shape().as_list() for u in self.U])

            with tf.name_scope('bilinear'):
                interpolated = reduce(bilinear, self.U, 1)

            final_shape = component_dims + shared_dims[:1]
            return tf.reshape(final_shape, interpolated, name='reshaped')

    def train_cp(self, X, optimizer, epochs=10):
        """
        TODO: describe

        """
        #optimizer = lambda cost, U: Optimizer(cost, train_vars=[U])
        X_var = tf.constant(X)
        train_ops = self.get_train_ops(X_var, optimizer)

        with tf.Session() as sess:
            for e in range(epochs):
                for train_op, loss in shuffled(train_ops):
                    _, loss = sess.run([train_op, loss], feed_dict={X_var : X})






















# ========================================================================================================

def unfold(X, shape, n):
    """
    Unfold a TF tensor of shape (d_1, d_2, ..., d_N)
        into a matrix   (d_n, D' )
        where    D' = d_1 * d_2 * ... * d_n-1 * d_n+1 * ... * d_N
    """
    s = shape[n]
    d1 = tf.prod(shape[:n])
    d2 = tf.prod(shape[(n+1):])

    X_ = tf.reshape(X, [d1, s, d2])
    return tf.reshape(X_, [s, d1*d2])   # TODO this and the line above it are likely to break

def init_nvecs(X, rank, do_flipsign=True):
    """
    Eigendecomposition of mode-n unfolding of a tensor

    TODO
    - will there be issues if we don't handle sparsity specially?
    """
    Y = tf.matmul(X, X.T)

    N = Y.shape[0]
    _, U = eigh(Y, )
    # reverse order of eigenvectors such that eigenvalues are decreasing
    U = array(U[:, ::-1])
    # flip sign
    if do_flipsign:
        U = flipsign(U)

    n_eigvals = (N - rank, N - 1)

    e,v = tf.self.adjoint_eig(X)
    return e





def tf_frobenius(X):
    return tf.reduce_sum(X ** 2) ** 0.5
