# coding: utf-8

import numpy as np
from scipy.linalg import hilbert
import matplotlib.pyplot as pl


def calculate_norm(a):
    norm = np.sqrt(np.sum(np.square(a)))
    return norm


def calculate_h(a):
    a = a.astype(np.complex)
    norm = calculate_norm(a)
    a_reflection = np.zeros_like(a).astype(np.complex)
    a_reflection[0] = norm
    if a[0] >= 0:
        v = a_reflection + a
    else:
        v = a - a_reflection
    is_all_zero = np.all((v == 0))
    if is_all_zero:
        v[0] = 1
        return np.identity(a.shape[0]).astype(np.complex), v
    v_k = v / calculate_norm(v)
    H = np.identity(a.shape[0]).astype(np.complex) - 2. * np.outer(v_k, np.conjugate(v_k))
    return H, v_k


def implicit_qr(A):
    W = A.astype(np.complex).copy()
    m, n = A.shape
    A = A.astype(np.complex)
    for k in range(A.shape[1]):
        a = A[k:m, k:n][:, 0]
        H, v_k = calculate_h(a)

        A[k:m, k:n] = np.matmul(H, A[k:m, k:n])

        W[k:m, k:n][:, 0] = v_k
        W[:k, k:n][:, 0] = 0
    return W, A


def form_q(W):
    m, n = W.shape
    Q = np.identity(W.shape[0]).astype(np.complex)
    for k in range(n):
        I = np.identity(W.shape[0]).astype(np.complex)
        v_k = W[k:m, k:n][:, 0]
        i = np.identity(I[k:m, k:m].shape[0]).astype(np.complex)
        I[k:m, k:m] = i - 2 * np.outer(v_k, np.conjugate(v_k))
        Q = np.matmul(I, Q)
    return Q.T


def tridiag(A):
    W = A.astype(np.complex).copy()
    m, n = A.shape
    A = A.astype(np.complex)
    # m==n since eigen values are computed on square symmetric matrices for our exercise
    for k in range(A.shape[1] - 2):
        a = A[k+1:m, k]
        H, v_k = calculate_h(a)

        A[k+1:m, k:n] = np.matmul(H, A[k+1:m, k:n])
        A[:m, k+1:n] = np.matmul(A[:m, k+1:n], H)

        W[k+1:m, k] = v_k
        W[:k+1, k] = 0
    return A


def qralg(T, run_with_driver=False):
    if not run_with_driver:
        T = tridiag(T)
    m, n = T.shape
    t = [np.abs(T[m - 1, m - 2])]
    T_n = T
    while True:
        # print(f'Iteration: {len(t)}')
        W, R = implicit_qr(T_n)
        Q = form_q(W)
        T_n = np.matmul(R, Q)
        crit = np.abs(T_n[m - 1, m - 2])
        t.append(crit)
        if crit < pow(10, -16):
            break
    return T_n, t


def wilkinson_shift(B):
    if B.size == 1:
        return B[0][0]
    delta = (B[0, 0] - B[1, 1]) / 2
    sign = np.sign(delta)
    if sign == 0:
        sign = 1

    shift = B[1, 1] - (sign * np.square(B[0, 1]) / (np.abs(delta) + np.sqrt(np.square(delta) + np.square(B[0, 1]))))
    return shift


def shifted_qralg(T, run_with_driver=False):
    if not run_with_driver:
        T = tridiag(T)
    m, n = T.shape
    t = [np.abs(T[m - 1, m - 2])]
    T_n = T
    while True:
        # print(f'Iteration: {len(t)}')
        shift = wilkinson_shift(T_n[m-2:, n-2:])
        W, R = implicit_qr(T_n - shift * np.identity(m))
        Q = form_q(W)
        T_n = np.matmul(R, Q) + shift * np.identity(m)
        crit = np.abs(T_n[m - 1, m - 2])
        t.append(crit)
        if crit < pow(10, -14):
            break
    return T_n, t


def qralg_driver(A, shift):
    m, n = A.shape
    all_t = []
    eigenvals = []

    A = tridiag(A)
    A_n = A
    while len(eigenvals) < m:
        if shift:
            A_n, t = shifted_qralg(A_n, run_with_driver=True)
        else:

            A_n, t = shifted_qralg(A_n, run_with_driver=True)

        m_k, n_k = A_n.shape
        eigenvals.append(A_n[m_k - 1, n_k - 1])
        A_n = A_n[:m_k - 1, : n_k - 1]

        if A_n.size == 1:
            eigenvals.append(A_n[0][0])
            break
        all_t.append(t)

    return eigenvals, all_t
