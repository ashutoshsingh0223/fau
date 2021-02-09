import numpy as np


def rayleigh(A, v0):
    eig_vec = v0
    ev = np.dot(np.dot(eig_vec.T, A), eig_vec)
    convergence_termination_count = 0
    while True:
        eig_vec = np.linalg.lstsq(A - ev * np.eye(len(A)).astype(np.complex), eig_vec, rcond=-1)[0]
        eig_vec = eig_vec / np.sqrt(np.sum(np.square(eig_vec)))
        ev_1 = np.dot(np.dot(eig_vec.T, A), eig_vec)
        diff = np.abs(ev_1 - ev)
        ev = ev_1
        if diff < 10e-14:
            convergence_termination_count += 1
            if convergence_termination_count >= 2:
                break
    return eig_vec, ev
