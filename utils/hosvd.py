'''Implementation of "A Multilinear Singualr Value Decomposition", Lieven De Lathauwer,
Bart De Moor and Joos Vandewalle, in Society for Industrial and Applied Mathematics, 2000'''

import numpy as np
# from numpy.linalg import svd
from scipy.linalg.interpolative import svd
import math

def unfold_column_pos(index, n, shape):
    trans = np.concatenate((np.arange(n, shape.shape[0]), np.arange(n)))
    reordered_index = index[trans]
    reordered_shape = shape[trans]
    column_pos = 0
    for i in range(1, reordered_index.shape[0]):
        column_pos += (reordered_index[i]*np.prod(reordered_shape[i+1:]))
    return (reordered_index[0], column_pos)

def unfold(A, n):
    return np.reshape(
        np.transpose(A, np.concatenate((np.arange(n, A.ndim), np.arange(n)))), [A.shape[n], -1])

def scalar_product(A, B):
    # A and B are scalars
    if A.ndim == 0:
        return A*B
    if A.ndim == 1:
        return np.dot(A, B)

    N = A.shape[0];
    res = 0
    for i in range(N):
        res += (scalar_product(A[i], B[i]))
    return res

def is_orthogonal(array1, array2):
    if(math.fabs(scalar_product(array1, array2)) <= 1e-9):
        return True
    else:
        return False

def tensor_Fnorm(A):
    return np.sqrt(scalar_product(A, A))

def my_svd(A, eps_or_k=0.01):
    if A.dtype != np.float64:
        A = A.astype(np.float64)
    U, S, V = svd(A, eps_or_k, rand=False)
    return U, S, V.T

def hosvd_S(A, U):
    d = len(U);
    S = A.copy()
    for i in range(d):
        S = np.tensordot(S, U[i], (0, 0))
    return S

def hosvd_cnst(S, U):
    A = S.copy()
    for i in range(len(U)):
        A = np.tensordot(A, U[i], (0, 1))
    return A

def hosvd_dcmp(A, eps_or_k=0.01):
    d = A.ndim
    s = A.shape
    max_rank = list(s)

    # print('eps_or_k = ', eps_or_k)
    # print('max_rank = ', max_rank)

    if isinstance(eps_or_k, list):
        if np.any(np.array(eps_or_k) > np.array(max_rank)):
            raise ValueError('the rank is up to %s' % str(max_rank))
    else:
        if eps_or_k > 1:
            raise ValueError('Epsilon is up to 1')
        if not isinstance(eps_or_k, list):
            eps_or_k = [eps_or_k] * d

    U = [my_svd(unfold(A.copy(), i), eps_or_k[i])[0] for i in range(d)]
    S = hosvd_S(A, U)
    return U, S


if __name__ == '__main__':
    A1 = np.array([[[1, 1, 0], [2, 2, 0]], [[1, -1, 2], [2, -2, 4]], [[2, 0, 2], [4, 0, 4]]])
    A1_unfold0 = unfold(A1, 0)
    A = np.random.rand(24).reshape((3, 2, 4))
    B = np.random.rand(24).reshape((3, 2, 4))
    A2 = np.array([[[0.9073, 0.7158, -0.3698],
      [1.7842, 1.697, 0.0151],
      [2.1236, -0.074, 1.4429]],
     [[0.8924, -0.4898, 2.4288],
      [1.7753, -1.5077, 4.0337],
      [-0.6631, 1.9103, -1.7495]],
     [[2.1488, 0.3054, 2.3753],
      [4.2495, 0.3207, 4.7146],
      [1.826, 2.1335, -0.2716]]])
    # A2_unfold0 = unfold(A2, 0)
    # A2_unfold1 = unfold(A2, 1)
    # A2_unfold2 = unfold(A2, 2)
    U, S = hosvd_dcmp(A2, [3, 3, 3])
    print(S)
    A2_recnst = hosvd_cnst(S, U)
    print(A2)