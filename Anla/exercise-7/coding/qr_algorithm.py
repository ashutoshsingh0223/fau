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
    # print("+++++++++++++++++++++++++++++++++++++")
    # print(A)
    return W, A


def form_q(W):
    Q = W.copy() #TODO
    return Q


def tridiag(A):
    pass #TODO
    return A


def qralg(T):
    t = []
    pass #TODO
    return (T, t)


def shifted_qralg(T):
    t = []
    pass #TODO
    return (T, t)


def qralg_driver(A, shift):
    all_t = []
    eigenvals = [] #TODO
    return (eigenvals, all_t)

