import numpy as np

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
        a = A[k:m, k]
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
