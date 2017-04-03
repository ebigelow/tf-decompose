
import numpy as np
import tensorflow as tf
from scipy.linalg import eigh


def shuffled(ls):
    return sorted(list(ls), key=lambda _: np.random.rand())

def nvecs(X, rank, n):
    """
    Initialize U as the top components of the left singular value of X unfolded along n.

    """
    X_ = unfold_np(X, n)
    Y = X_.dot(X_.T)
    N = Y.shape[0]

    _, U = eigh(Y, eigvals=(N - rank, N - 1))
    # reverse order of eigenvectors such that eigenvalues are decreasing
    U = np.array(U[:, ::-1])
    return U

def get_fit(X, Y):
    """
    Compute squared frobenius distance between 2 numpy matrices.
        ||X - Y||_F^2  = <X,X> + <Y,Y> - 2 <X,Y>

    """
    normX = (X ** 2).sum()
    normY = (Y ** 2).sum()
    inner = (X * Y).sum()

    normresidual = normX  +  normY - 2*inner
    return 1 - (normresidual / normX)

def unfold_np(arr, ax):
    """
    Unfold a numpy tensor along its nth axis.
      from: https://gist.github.com/nirum/79d8e14da106c77c02c1

    """
    return np.rollaxis(arr, ax, 0).reshape(arr.shape[ax], -1)

def unfold_tf(A, n):
    """
    Unfold a TF tensor A along its nth axis.

    input shape  : (d_1, d_2, ..., d_N)
    output shape : (d_n, D' )
    where    D' = d_1 * d_2 * ... * d_n-1 * d_n+1 * ... * d_N

    """
    shape = A.get_shape().as_list()
    idxs = [i for i,_ in enumerate(shape)]

    new_idxs = [n] + idxs[:n] + idxs[(n+1):]
    B = tf.transpose(A, new_idxs)

    dim = shape[n]
    return tf.reshape(B, [dim, -1])

def refold_tf(A, shape, n):
    """
    Refold an unrolled tensor.

    Arguments
    ---------
    A (tf.Variable) : unrolled tensor
    shape (list) : list of integers specifying output tensor's shape
    n (int) : assume A is unrolled along shape[n]

    """
    idxs = [i for i,_ in enumerate(shape)]

    shape_temp = [shape[n]] + shape[:n] + shape[(n+1):]
    B = tf.reshape(A, shape_temp)

    new_idxs = idxs[1:(n+1)] + [0] + idxs[(n+1):]
    return tf.transpose(B, new_idxs)

def bilinear(A, B):
    """
    Return the bilinear tensor product of two TensorFlow tensors A,B.

    Note that A,B must share their final dimension, all other
      dimensions are used for the bilinear product.

    Tensor shapes (where i and j may refer to lists of indices):
          A      B           C
        (i,k)  (j,k)  =>  (i,j,k)

    Compute:
        C_ijk = A_ik    *  B_jk      (i,j,k index of C is equal to A_ik times B_jk )
              = A'_ijk  *  B'_ijk    (tiling A,B along along axes i,j respectively)
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
