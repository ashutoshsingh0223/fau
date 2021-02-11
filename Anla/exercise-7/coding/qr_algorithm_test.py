import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import hilbert
import unittest

import qr_algorithm as submission


class QRTest(unittest.TestCase):
    def check_tridiag(self, A):
        with self.subTest(A=A):
            sol = submission.tridiag(A.copy())
            assert_allclose(sol, np.triu(sol, -1), atol=1e-15,
                            err_msg='solution is not upper triangular')
            assert_allclose(sol, np.tril(sol, 1), atol=1e-15,
                            err_msg='solution is not lower triangular')

    def test_tridiag(self):
        A = hilbert(4)
        self.check_tridiag(A + A.T)

    def check_qr(self, A):
        with self.subTest(A=A):
            sol = submission.qralg(A.copy())[0]
            m, n = A.shape

            self.assertLess(np.abs(sol[m - 1, m - 2]), 1e-12,
                            msg='not fully converged')

    def test_qr(self):
        A = hilbert(4)
        self.check_qr(A + A.T)

    def check_qr_shifted(self, A):
        with self.subTest(A=A):
            sol = submission.shifted_qralg(A.copy())[0]
            m, n = A.shape

            self.assertLess(np.abs(sol[m - 1, m - 2]), 1e-12,
                            msg='not fully converged')

    def test_qr_shifted(self):
        A = hilbert(4)
        self.check_qr(A + A.T)

    def check_qr_driver(self, A):
        with self.subTest(A=A):
            sol = np.array(submission.qralg_driver(A.copy(), False)[0])
            eig_vals = np.linalg.eigvals(A)
            for eig in eig_vals:
                self.assertLess(np.abs(sol - eig).min(), 1e-14,
                                msg=f'eigenvalue {eig} nof found')
            for eig in sol:
                self.assertLess(np.abs(eig_vals - eig).min(), 1e-14,
                                msg=f'{eig} is not an eigenvalue')

            sol = np.array(submission.qralg_driver(A.copy(), True)[0])
            eig_vals = np.linalg.eigvals(A)
            for eig in eig_vals:
                self.assertLess(np.abs(sol - eig).min(), 1e-14,
                                msg=f'eigenvalue {eig} not found')
            for eig in sol:
                self.assertLess(np.abs(eig_vals - eig).min(), 1e-14,
                                msg=f'{eig} is not an eigenvalue')

    def test_qr_driver(self):
        A = hilbert(4)
        self.check_qr_driver(A + A.T)


if __name__ == '__main__':
    unittest.main(verbosity=2)
