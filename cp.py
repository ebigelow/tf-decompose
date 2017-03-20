import logging
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

def khatri_rao(components, name='reconstructed'):
    """
    khatri-rao product maps
    (i,k) . (j,k) --> (i,j,k)

    To do this, we first compute a bilinear tensor product
    (i,k) . (j,k) --> (i*j, k)

    Then must reshape this
    (i*j, k) --> (i,j,k)


    goal     (i,l) . (j,l) . (k,l) --> (i,j,k,l)
    step 1   (i,l) . (j,l) . (k,l) --> (i*j*k, l)
    step 2   (i*j*k, l) --> (i,j,k,l)

    """
    with tf.name_scope(name):
        component_dims, shared_dims = zip(*[U.get_shape().as_list() for U in components])
        if not all([ (s == shared_dims[0]) for s in shared_dims ]):
            raise ValueError('Error, shared dimension mismatch %s' % str(shared_dims))

        new_dim = np.prod(component_dims)
        # TODO how to do this bilinear product ?!?!?!
        bilinear = tf.Variable([new_dim, shared_dims[0]], name='bilinear')


        final_shape = component_dims + [shared_dims[0]]
        kr_prod = tf.reshape(final_shape, bilinear, name='khatri_rao')
        return kr_prod



def naive_cp(X, TODO):
    """
    set X* = sum_r a_1 . a_2 . a_3 ...
    """

    U = TODO_init_somehow()

    X_predict = khatri_rao(U, name='X_predict')

    loss = tf.reduce_sum(





# -----------------------------------







class KruskalTensor:
    """
    Used for CP decomposition of a tensor.

    There will be a little messiness at some points because tf.einsum doesn't allow
      sloppy formatting, e.g. `tf.einsum('i...', '...i')`

    """
    def __init__(self, shape, rank, init='random', dtype=tf.float32):
        self.shape = shape
        self.order = len(shape)
        self.rank = rank
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

    def cp_als(self, X, fit_method='full',
               max_iter=500, converged=1e-5):
        """
        Learn [U_1, ... U_N] ~~ X  using the alternating least squares' algorithm.

        """
        U = self.U.copy()

        norm_X = tf.reduce_sum(X ** 2) ** 0.5

        fit = 0
        exectimes = []

        for itr in range(max_iter):
            for n in range(self.order):

                U_new = uttkrp(X, U, n)
                Y = tf.ones((self.rank, self.rank), dtype=self.dtype)

                unroll_orders = (list(range(n)) + list(range(n + 1, self.order)))
                for i in unroll_orders:
                    Y = Y * tf.matmul(U[i].T, U[i])

                U_new = tf.matmul(U_new, tf.pinv(Y))

                # Normalize
                if itr == 0:
                    lamb = tf.reduce_sum(U_new ** 2, axis=0) ** 0.5
                else:
                    lamb = tf.reduce_max(U_new, axis=0)
                    lamb[lamb < 1] = 1
                self.U[n] = U_new / lamb

            if fit_method == 'full':
                normresidual = norm_X ** 2 + self.norm() ** 2 - 2 * self.inner_prod(X)
                fit = 1 - (normresidual / norm_X ** 2)
            else:
                fit = itr

            if itr > 0 and fitchange < converged:
                break
        return U, lamb, fit, itr, array(exectimes)

    def norm(self):
        """
        Efficient computation of the Frobenius norm for ktensors

        """
        N = len(self.shape)
        coef = outer(self.lmbda, self.lmbda)
        for i in range(N):
            coef = coef * dot(self.U[i].T, self.U[i])
        return np.sqrt(coef.sum())

    def inner_prod(self, X):
        """
        Efficient computation of the inner product of a ktensor with another tensor

        """
        N = len(self.shape)
        R = len(self.lmbda)
        res = 0
        for r in range(R):
            vecs = []
            for n in range(N):
                vecs.append(self.U[n][:, r])
            res += self.lmbda[r] * X.ttv(tuple(vecs))
        return res





def uttkrp(X, N, U, n):
    """
    Unfolded tensor times Khatri-Rao product:
    :math:`M = \\unfold{X}{3} (U_1 \kr \cdots \kr U_N)`
    Computes the _matrix_ product of the unfolding
    of a tensor and the Khatri-Rao product of multiple matrices.
    Efficient computations are perfomed by the respective
    tensor implementations.
    Parameters
    ----------
    U : list of tensors
        Matrices for which the Khatri-Rao product is computed and
        which are multiplied with the tensor in mode ``mode``.
    mode : int
        Mode in which the Khatri-Rao product of ``U`` is multiplied
        with the tensor.
    Returns
    -------
    M : tensor
        Matrix which is the result of the matrix product of the unfolding of
        the tensor and the Khatri-Rao product of ``U``

    """
    order = list(range(n)) + list(range(n + 1, N))
    Z = khatri_rao(tuple(U[i] for i in order), reverse=True)
    return unfold(X, n).dot(Z)


def khatri_rao(Xs):
    """
    Xs : list of N tensors, sharing second dim. (rank in our case) but not first

    return tensor of shape
        ( d_1 * d_2 * ... * d_N    x    rank )

    TODO: get bilinear product code from detect-relationships

    """
    return 'TODO'


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
