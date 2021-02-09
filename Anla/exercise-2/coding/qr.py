import numpy as np




from math import sqrt
from pprint import pprint


def cmp(a, b):
    return (a > b) - (a < b)

def mult_matrix(M, N):
    """Multiply square matrices of same dimension M and N"""
    # Converts N into a list of tuples of columns
    tuple_N = zip(*N)

    # Nested list comprehension to calculate matrix multiplication
    return [[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in tuple_N] for row_m in M]

def trans_matrix(M):
    """Take the transpose of a matrix."""
    n = len(M)
    return [[ M[i][j] for i in range(n)] for j in range(n)]

def norm(x):
    """Return the Euclidean norm of the vector x."""
    return sqrt(sum([x_i**2 for x_i in x]))

def Q_i(Q_min, i, j, k):
    """Construct the Q_t matrix by left-top padding the matrix Q
    with elements from the identity matrix."""
    if i < k or j < k:
        return float(i == j)
    else:
        return Q_min[i-k][j-k]

def householder(A):
    """Performs a Householder Reflections based QR Decomposition of the
    matrix A. The function returns Q, an orthogonal matrix and R, an
    upper triangular matrix such that A = QR."""
    n = len(A)

    # Set R equal to A, and create Q as a zero matrix of the same size
    R = A
    Q = [[0.0] * n for i in range(n)]

    # The Householder procedure
    for k in range(n-1):  # We don't perform the procedure on a 1x1 matrix, so we reduce the index by 1
        # Create identity matrix of same size as A
        I = [[float(i == j) for i in range(n)] for j in range(n)]

        # Create the vectors x, e and the scalar alpha
        # Python does not have a sgn function, so we use cmp instead
        x = [row[k] for row in R[k:]]
        e = [row[k] for row in I[k:]]
        alpha = -cmp(x[0], 0) * norm(x)

        # Using anonymous functions, we create u and v
        u = map(lambda p,q: p + alpha * q, x, e)
        norm_u = norm(u)
        v = map(lambda p: p/norm_u, u)

        # Create the Q minor matrix
        Q_min = [ [float(i==j) - 2.0 * v[i] * v[j] for i in range(n-k)] for j in range(n-k) ]

        # "Pad out" the Q minor matrix with elements from the identity
        Q_t = [[ Q_i(Q_min,i,j,k) for i in range(n)] for j in range(n)]

        # If this is the first run through, right multiply by A,
        # else right multiply by Q
        if k == 0:
            Q = Q_t
            R = mult_matrix(Q_t,A)
        else:
            Q = mult_matrix(Q_t,Q)
            R = mult_matrix(Q_t,R)

    # Since Q is defined as the product of transposes of Q_t,
    # we need to take the transpose upon returning it
    return trans_matrix(Q), R




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

    # W = A.astype(np.complex).copy()
    # m, n = A.shape
    # A = A.astype(np.complex)
    # for k in range(A.shape[1]):
    #     a = A[k:m, k:n][:, 0]
    #     H, v_k = calculate_h(a)
    #
    #     A[k:m, k:n] = np.matmul(H, A[k:m, k:n])
    #
    #     W[k:m, k:n][:, 0] = v_k
    #     W[:k, k:n][:, 0] = 0
    # # print("+++++++++++++++++++++++++++++++++++++")
    # # print(A)
    Q, R = householder(A)
    return Q, R


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
