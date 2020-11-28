import numpy as np
from numpy.testing import assert_allclose
import unittest

import qr as submission


class QRTest(unittest.TestCase):
    def check_house(self, A):
        with self.subTest(A=A):
            W_ans, R_ans = submission.implicit_qr(A.copy())
            self.assertTupleEqual(R_ans.shape, A.shape,
                                  msg='R has a different shape than A')
            self.assertEqual(W_ans.shape[0], W_ans.shape[1],
                             msg='W is not quadratic')
            assert_allclose(R_ans, np.triu(R_ans), atol=1e-15,
                            err_msg='R is not upper triangular')
            assert_allclose(np.diag(np.conjugate(W_ans.T) @ W_ans), np.ones(W_ans.shape[0]), atol=1e-15,
                            err_msg='W is not unitary')

    # def test_house(self):
    #     A = np.eye(3)
    #     self.check_house(A)
    #
    #     A = np.diag([3, 2, 1])
    #     self.check_house(A)
    #
    #     A = np.diag([2, 1, 3])
    #     self.check_house(A)
    #
    #     A = np.array([[0, 2], [4, 0], [1, 0]])
    #     self.check_qr(A)
    #
    #     A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    #     self.check_house(A)
    #
    #     A = np.array([[0, 0, 3.0], [0, 4.0, 0], [5.0, 0, 0.0]])
    #     self.check_house(A)
    #
    #     A = np.array([[0, 0, 3j], [0, 4j, 0], [5j, 0, 0.0]])
    #     self.check_house(A)

    def check_form_q(self, W):
        with self.subTest(W=W):
            Q_ans = submission.form_q(W.copy())
            assert_allclose(Q_ans.T @ Q_ans, np.eye(Q_ans.shape[0]), atol=1e-14,
                            err_msg='Q is not unitary')

    def test_form_q(self):
        W = np.eye(3)
        self.check_form_q(W)

        W = np.array([[np.sqrt(2)*0.5, 0, 0], [0, 1, 0], [np.sqrt(2)*0.5, 0, -1]])
        self.check_form_q(W)

    def check_qr(self, A):
        with self.subTest(A=A):
            W_ans, R_ans = submission.implicit_qr(A.copy())
            Q_ans = submission.form_q(W_ans)
            assert_allclose(Q_ans @ R_ans, A, atol=1e-14,
                            err_msg='QR is not equal to A')

    def test_qr(self):
        A = np.eye(3)
        self.check_qr(A)

        A = np.diag([3, 2, 1])
        self.check_qr(A)

        A = np.diag([2, 1, 3])
        self.check_qr(A)

        A = np.array([[0, 2], [4, 0], [1, 0]])
        self.check_qr(A)

        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        self.check_qr(A)

        A = np.array([[0, 0, 3.0], [0, 4.0, 0], [5.0, 0, 0.0]])
        self.check_qr(A)

        A = np.array([[0, 0, 3j], [0, 4j, 0], [5j, 0, 0.0]])
        self.check_qr(A)


if __name__ == '__main__':
    unittest.main(verbosity=2)
