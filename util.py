import numpy as np
import numpy.lib.scimath as csqrt
import scipy 
from sklearn.utils import check_random_state
from scipy import fftpack
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import image
from scipy import sparse
from cmath import log,sqrt
from functools import partial
from sklearn.utils.extmath import safe_sparse_dot

def compute_rqb(A, rank, oversample=20, n_subspace=2, n_blocks=1, sparse=False,random_state=None):
    if n_blocks > 1:
        m, n = A.shape

        # index sets
        row_sets = np.array_split(range(m), n_blocks)

        Q_block = []
        K = []

        nblock = 1
        for rows in row_sets:
            # converts A to array, raise ValueError if A has inf or nan
            Qtemp, Ktemp = _compute_rqb(np.asarray_chkfinite(A[rows, :]), 
                rank=rank, oversample=oversample, n_subspace=n_subspace, 
                sparse=sparse, random_state=random_state)

            Q_block.append(Qtemp)
            K.append(Ktemp)
            nblock += 1

        Q_small, B = _compute_rqb(
            np.concatenate(K, axis=0), rank=rank, oversample=oversample,
            n_subspace=n_subspace, sparse=sparse, random_state=random_state)

        Q_small = np.vsplit(Q_small, n_blocks)

        Q = [Q_block[i].dot(Q_small[i]) for i in range(n_blocks)]
        Q = np.concatenate(Q, axis=0)

    else:
        Q, B = _compute_rqb(np.asarray_chkfinite(A), 
            rank=rank, oversample=oversample, n_subspace=n_subspace,
            sparse=sparse, random_state=random_state)

    return Q, B

def _compute_rqb(A, rank, oversample, n_subspace, sparse, random_state):
    if sparse:
        Q = sparse_johnson_lindenstrauss(A, rank + oversample,
                                         random_state=random_state)
    else:
        Q = johnson_lindenstrauss(A, rank + oversample, random_state=random_state)

    if n_subspace > 0:
        Q = perform_subspace_iterations(A, Q, n_iter=n_subspace, axis=1)
    else:
        Q = orthonormalize(Q)

    # Project the data matrix a into a lower dimensional subspace
    B = conjugate_transpose(Q).dot(A)

    return Q, B


def random_gaussian_map(A, l, axis, random_state):
    """generate random gaussian map"""
    return random_state.standard_normal(size=(A.shape[axis], l)).astype(A.dtype)
def orthonormalize(A, overwrite_a=True, check_finite=False):
    """orthonormalize the columns of A via QR decomposition"""
    # NOTE: for A(m, n) 'economic' returns Q(m, k), R(k, n) where k is min(m, n)
    # TODO: when does overwrite_a even work? (fortran?)
    Q, _ = la.qr(A, overwrite_a=overwrite_a, check_finite=check_finite,
                     mode='economic', pivoting=False)
    return Q

def perform_subspace_iterations(A, Q, n_iter=2, axis=1):
    """perform subspace iterations on Q"""
    # TODO: can we figure out how not to transpose for row wise
    if axis == 0:
        Q = Q.T

    # orthonormalize Y, overwriting
    Q = orthonormalize(Q)

    # perform subspace iterations
    for _ in range(n_iter):
        if axis == 0:
            Z = orthonormalize(A.dot(Q))
            Q = orthonormalize(A.T.dot(Z))
        else:
            Z = orthonormalize(A.T.dot(Q))
            Q = orthonormalize(A.dot(Z))

    if axis == 0:
        return Q.T
    return Q

def conjugate_transpose(A):
    """Performs conjugate transpose of A"""
    if A.dtype == np.complexfloating:
        return A.conj().T
    return A.T

def sparse_random_map(A, l, axis, density, random_state):
    """generate sparse random sampling"""
    # TODO: evaluete random_state paramter: we want to pass it to sparse.random
    #       most definitely to sparsely sample the nnz elements of Omega, but
    #       is using random_state in data_rvs redundant?
    values = (-sqrt(1. / density), sqrt(1. / density))
    data_rvs = partial(random_state.choice, values)

    return sparse.random(A.shape[axis], l, density=density, data_rvs=data_rvs,
                         random_state=random_state, dtype=A.dtype)


def randomized_uniform_sampling(A, l, axis=1, random_state=None):
    """Uniform randomized sampling transform.

    Given an m x n matrix A, and an integer l, this returns an m x l
    random subset of the range of A.

    """
    random_state = check_random_state(random_state)
    A = np.asarray(A)

    # sample l rows/columns with equal probability
    idx = random_axis_sample(A, l, axis, random_state)

    return np.take(A, idx, axis=axis)

def random_axis_sample(A, l, axis, random_state):
    """randomly sample the index of axis"""
    return random_state.choice(A.shape[axis], size=l, replace=False)

def johnson_lindenstrauss(A, l, axis=1, random_state=None):
    """
    Given an m x n matrix A, and an integer l, this scheme computes an m x l
    orthonormal matrix Q whose range approximates the range of A
    """
    random_state = check_random_state(random_state)

    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('A must be a 2D array, not %dD' % A.ndim)

    if axis not in (0, 1):
        raise ValueError('If supplied, axis must be in (0, 1)')

    # construct gaussian random matrix
    Omega =random_gaussian_map(A, l, axis, random_state)

    # project A onto Omega
    if axis == 0:
        return Omega.T.dot(A)
    return A.dot(Omega)

def fast_johnson_lindenstrauss(A, l, axis=1, random_state=None):
    """

    Given an m x n matrix A, and an integer l, this scheme computes an m x l
    orthonormal matrix Q whose range approximates the range of A

    """
#     m,n=A.shape
    random_state = check_random_state(random_state)

    A = np.asarray_chkfinite(A)
    if A.ndim != 2:
        raise ValueError('A must be a 2D array, not %dD' % A.ndim)

    if axis not in (0, 1):
        raise ValueError('If supplied, axis must be in (0, 1)')

    # TODO: Find name for sketch and put in _sketches
    # construct gaussian random matrix
    diag = random_state.choice((-1, 1), size=A.shape[axis]).astype(A.dtype)

    if axis == 0:
        diag = diag[:, np.newaxis]

    # discrete fourier transform of AD (or DA)
    FDA = fftpack.dct(A * diag, axis=axis, norm='ortho')

    # randomly sample axis
    return randomized_uniform_sampling(
        FDA, l, axis=axis, random_state=random_state)

def sparse_johnson_lindenstrauss(A, l, density=None, axis=1, random_state=None):
    """

    Given an m x n matrix A, and an integer l, this scheme computes an m x l
    orthonormal matrix Q whose range approximates the range of A

    Parameters
    ----------
    density : sparse matrix density

    """
    random_state = check_random_state(random_state)

    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('A must be a 2D array, not %dD' % A.ndim)

    if axis not in (0, 1):
        raise ValueError('If supplied, axis must be in (0, 1)')

    if density is None:
        density = log(A.shape[0]) / A.shape[0]
        density=np.real(density)

    # construct sparse sketch
    Omega =sparse_random_map(A, l, axis, density, random_state)

    # project A onto Omega
    if axis == 0:
        return safe_sparse_dot(Omega.T, A)
    return safe_sparse_dot(A, Omega)