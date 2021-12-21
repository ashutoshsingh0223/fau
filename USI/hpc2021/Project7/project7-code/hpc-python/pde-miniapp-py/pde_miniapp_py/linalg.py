"""
Collection of linear algebra operations and CG solver
"""
from mpi4py import MPI
import numpy as np
import math
from . import data
from . import operators

def hpc_dot(x, y):
    """Computes the inner product of x and y"""
    # prod = 0
    # x_vec = x.inner[...][0]
    # y_vec = y.inner[...][0]
    # for i in range(len(x_vec)):
    #     prod += x_vec[i] * y_vec[i]

    dot_prod = np.sum(x.inner[...] *  y.inner[...])
    return dot_prod

def hpc_norm2(x):
    """Computes the 2-norm of x"""
    # norm2 = 0
    # x_vec = x.inner[...].flatten()
    # for i in range(len(x_vec)):
    #     norm2 += x_vec[i] * x_vec[i]
    # norm2 = math.sqrt(norm2)


    norm2 = np.sqrt(np.sum(np.square(x.inner)))

    return norm2

class hpc_cg:
    """Conjugate gradient solver class: solve the linear system A x = b"""
    def __init__(self, domain):
        self._Ap = data.Field(domain)
        self._r  = data.Field(domain)
        self._p  = data.Field(domain)

        self._xold  = data.Field(domain)
        self._v  = data.Field(domain)
        self._Fxold  = data.Field(domain)
        self._Fx  = data.Field(domain)
        self._v  = data.Field(domain)

    def solve(self, A, b, x, tol, maxiter):
        """Solve the linear system A x = b"""
        # initialize
        A(x, self._Ap)
        self._r.inner[...] = b.inner[...] - self._Ap.inner[...]
        self._p.inner[...] = self._r.inner[...]
        delta_kp = hpc_dot(self._r, self._r)
        # print(self._r.inner.shape)
        # print(b.inner.shape)
 
        # iterate
        converged = False
        for k in range(0, maxiter):
            if k % 50 == 0:
                print(f'CG Iteration = {k}')

            delta_k = delta_kp
            if delta_k < tol**2:
                converged = True
                break
            A(self._p, self._Ap)
            alpha = delta_k/hpc_dot(self._p, self._Ap)
            x.inner[...] += alpha*self._p.inner[...]
            self._r.inner[...] -= alpha*self._Ap.inner[...]
            delta_kp = hpc_dot(self._r, self._r)
            self._p.inner[...] = ( self._r.inner[...]
                                  + delta_kp/delta_k*self._p.inner[...] )

        return converged, k + 1

