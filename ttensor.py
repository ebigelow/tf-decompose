
import tensorflow as tf
from dtensor import DecomposedTensor



def unfold(A, n):
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


def refold(A, shape, n):
    idxs = [i for i,_ in enumerate(shape)]

    shape_temp = [shape[n]] + shape[:n] + shape[(n+1):]
    B = tf.reshape(A, shape_temp)

    new_idxs = idxs[1:(n+1)] + [0] + idxs[(n+1):]
    return tf.transpose(B, new_idxs)




class TuckerTensor(DecomposedTensor):
    """
    Used for Tucker decomposition of a tensor.

    """
    def __init__(self, shape, ranks, regularize=1e-5, init='random', dtype=tf.float64):
        self.shape = shape
        self.order = len(shape)
        self.ranks = ranks if (type(ranks) is list) else [ranks]*self.order
        self.regularize = regularize
        self.dtype = dtype
        self.init_random()
        self.init_reconstruct()
        self.init_norm()

    def init_random(self, a=0.0, b=1.0):
        self.G = tf.Variable(tf.random_uniform(
            self.ranks, a, b, self.dtype), name='G')

        with tf.name_scope('U'):
            self.U = [None] * self.order
            for n in range(0, self.order):
                shape = (self.shape[n], self.ranks[n])
                self.U[n] = tf.Variable(tf.random_uniform(
                    shape, minval=a, maxval=b, dtype=self.dtype), name=str(n))

    def init_reconstruct(self):
        """
        Initialize variable for reconstructed tensor `X` with components `U`.

        """
        G_to_X = self.G
        shape = self.ranks[:]

        for n in range(self.order):
            shape[n] = self.shape[n]
            name = None if (n < self.order-1) else 'X'

            Un_mul_G = tf.matmul(self.U[n], unfold(G_to_X, n))
            with tf.name_scope(name):
                G_to_X = refold(Un_mul_G, shape, n)

        self.X = G_to_X

    def init_norm(self):
        pass
