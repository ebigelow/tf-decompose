import logging
import time
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

C  0  B     (R x N x P)


"""



# super naive way


def outer(U, V):
    """
    outer tensor product of two arbitrarily dimension tensors

    (i,j,k) . (l,m)  -->  (i,j,k,l,m)

    """
    U_ = tf.expand_dims(U, -1)
    V_ = tf.expand_dims(V,  0)
    return tf.matmul(U_, V_)

def naive_cp(X, TODO):
    """
    set X* = sum_r a_1 . a_2 . a_3 ...

    """
    loss = tf.reduce_sum(X)



# -----------------------------------



def als(X, rank, ainit='nvecs', maxinit=500,
        fit_method='full', conv=1e-5, dtype=np.float):
    """
    X : tensor_mixin
        The tensor to be decomposed.
    rank : int
        Tensor rank of the decomposition.
    init : {'random', 'nvecs'}, optional
        The initialization method to use.
            - random : Factor matrices are initialized randomly.
            - nvecs : Factor matrices are initialzed via HOSVD.
        (default 'nvecs')
    max_iter : int, optional
        Maximium number of iterations of the ALS algorithm.
        (default 500)
    fit_method : {'full', None}
        The method to compute the fit of the factorization
            - 'full' : Compute least-squares fit of the dense approximation of.
                       X and X.
            - None : Do not compute the fit of the factorization, but iterate
                     until ``max_iter`` (Useful for large-scale tensors).
        (default 'full')
    conv : float
        Convergence tolerance on difference of fit between iterations
        (default 1e-5)
    Returns
    -------
    P : ktensor
        Rank ``rank`` factorization of X. ``P.U[i]`` corresponds to the factor
        matrix for the i-th mode. ``P.lambda[i]`` corresponds to the weight
        of the i-th mode.
    fit : float
        Fit of the factorization compared to ``X``
    itr : int
        Number of iterations that were needed until convergence
    exectimes : ndarray of floats
        Time needed for each single iteration

    """
    N = X.ndim
    normX = norm(X)

    U = _init(ainit, X, N, rank, dtype)
    fit = 0
    exectimes = []
    for itr in range(maxiter):
        tic = time.clock()
        fitold = fit

        for n in range(N):
            Unew = X.uttkrp(U, n)
            Y = ones((rank, rank), dtype=dtype)
            for i in (list(range(n)) + list(range(n + 1, N))):
                Y = Y * dot(U[i].T, U[i])
            Unew = Unew.dot(pinv(Y))
            # Normalize
            if itr == 0:
                lmbda = sqrt((Unew ** 2).sum(axis=0))
            else:
                lmbda = Unew.max(axis=0)
                lmbda[lmbda < 1] = 1
            U[n] = Unew / lmbda

        # P = ktensor(U, lmbda)
        if fit_method == 'full':
            normresidual = normX ** 2 + P.norm() ** 2 - 2 * P.innerprod(X)
            fit = 1 - (normresidual / normX ** 2)
        else:
            fit = itr
        fitchange = abs(fitold - fit)
        exectimes.append(time.clock() - tic)
        _log.debug(
            '[%3d] fit: %.5f | delta: %7.1e | secs: %.5f' %
            (itr, fit, fitchange, exectimes[-1])
        )
        if itr > 0 and fitchange < conv:
            break

    return P, fit, itr, array(exectimes)




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
        self.init = init
        self.fit_method = fit_method
        self.max_iter = max_iter
        self.converged = converged
        self.dtype = dtype
        self.init_random()

    def init_random(self, a=0.0, b=1.0):
        """
        Init component matrices `U` with random vals in the interval [a,b).

        """
        self.U = [] * self.order
        for n in range(0, self.order):
            shape = (X.shape[n], rank)
            self.U[n] = tf.random_uniform(shape, name=('U_%d' % n), minval=a, maxval=b)

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
                    lamb = sqrt((U_new ** 2).sum(axis=0))
                else:
                    lamb = U_new.max(axis=0)
                    lamb[lamb < 1] = 1
                self.U[n] = U_new / lamb

            if fit_method == 'full':
                normresidual = norm_X ** 2 + P.norm() ** 2 - 2 * P.inner_prod(X)
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
