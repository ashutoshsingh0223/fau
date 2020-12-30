import numpy as np


def lu(A):
    n = len(A)
    U = A.astype(np.complex)
    L = np.eye(n).astype(np.complex)
    for k in range(n - 1):
        for j in range(k + 1, n):
            l_j_k = U[j, k] / U[k, k]
            U[j, k:] = U[j, k:] - l_j_k * U[k, k:]
            L[j, k] = l_j_k
    return (L, U)


def lu_complete(A):
    n = len(A)
    U = A.astype(np.complex)
    L = np.eye(n).astype(np.complex)
    P = np.eye(n)
    Q = np.eye(n)
    for k in range(n - 1):
        max_index = np.unravel_index(U[k:, k:].argmax(), U[k:, k:].shape)
        ii = max_index[0] + k
        jj = max_index[1] + k

        P[[k, ii]] = P[[ii, k]]
        Q[:, [k, jj]] = Q[:, [jj, k]]

        U[[k, ii], k:] = U[[ii, k], k:]
        U[k:, [k, jj]] = U[k:, [jj, k]]

        L[[k, ii], :k] = L[[ii, k], :k]
        L[:k, [k, jj]] = L[:k, [jj, k]]

        for j in range(k + 1, n):
            l_j_k = U[j, k] / (U[k, k] + np.finfo(float).eps)
            if l_j_k == 0:
                break
            U[j, k:] = U[j, k:] - l_j_k * U[k, k:]
            L[j, k] = l_j_k

    return (P, Q, L, U)
