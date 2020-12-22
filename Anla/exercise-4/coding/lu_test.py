import numpy as np
from numpy.testing import assert_allclose
import unittest

import lu as submission


class LUTest(unittest.TestCase):
    def check_lu(self, A):
        with self.subTest(A=A):
            L_ans, U_ans = submission.lu(A.copy())
            self.assertTupleEqual(L_ans.shape, A.shape,
                                  msg='L has a different shape than A')
            self.assertTupleEqual(U_ans.shape, A.shape,
                                  msg='U has a different shape than A')
            assert_allclose(L_ans, np.tril(L_ans), atol=1e-15,
                            err_msg='L is not lower triangular')
            assert_allclose(U_ans, np.triu(U_ans), atol=1e-15,
                            err_msg='U is not upper triangular')
            assert_allclose(A, L_ans @ U_ans, atol=1e-10,
                            err_msg='A != LU')

    def test_lu(self):
        A = np.eye(3)
        self.check_lu(A)

        A = np.diag([3, 2, 1])
        self.check_lu(A)

        A = np.diag([2, 1, 3])
        self.check_lu(A)

        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        self.check_lu(A)

    def check_lu_complete(self, A):
        with self.subTest(A=A):
            P_ans, Q_ans, L_ans, U_ans = submission.lu_complete(A.copy())
            self.assertTupleEqual(P_ans.shape, A.shape,
                                  msg='P has a different shape than A')
            self.assertTupleEqual(Q_ans.shape, A.shape,
                                  msg='Q has a different shape than A')
            self.assertTupleEqual(L_ans.shape, A.shape,
                                  msg='L has a different shape than A')
            self.assertTupleEqual(U_ans.shape, A.shape,
                                  msg='U has a different shape than A')
            assert_allclose(P_ans @ P_ans.T, np.eye(P_ans.shape[0]), atol=1e-15,
                            err_msg='P is not orthogonal')
            assert_allclose(Q_ans @ Q_ans.T, np.eye(Q_ans.shape[0]), atol=1e-15,
                            err_msg='Q is not orthogonal')
            assert_allclose(L_ans, np.tril(L_ans), atol=1e-15,
                            err_msg='L is not lower triangular')
            assert_allclose(U_ans, np.triu(U_ans), atol=1e-15,
                            err_msg='U is not upper triangular')
            assert_allclose(A, P_ans.T @ L_ans @ U_ans @ Q_ans.T, atol=1e-10,
                            err_msg='PAQ != LU')

    def test_lu_complete(self):
        A = np.eye(3)
        self.check_lu_complete(A)

        A = np.diag([3, 2, 1])
        self.check_lu_complete(A)

        A = np.diag([2, 1, 3])
        self.check_lu_complete(A)

        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        self.check_lu_complete(A)


if __name__ == '__main__':
    unittest.main(verbosity=2)
