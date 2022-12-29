import numpy as np
from melodion.ops import psi, vec, mat

def sample_matrices():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    Y = np.array([[7, 8, 9], [10, 11, 12]])
    return X, Y


def test_psi():
    X, Y = sample_matrices()
    d = psi(X, Y)
    assert np.all(d == [50, 167])


def test_vec():
    X, Y = sample_matrices()
    xv = vec(X)
    assert np.all(xv == [1, 2, 3, 4, 5, 6])


def test_mat():
    X, Y = sample_matrices()
    matrix_shape = np.shape(X)
    xv = vec(X)
    Xx = mat(xv, matrix_shape)
    assert np.all(X == Xx)


if __name__ == "__main__":
    test_psi()
    test_vec()
    test_mat()