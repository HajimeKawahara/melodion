import numpy as np

def psi(X, Y):
    """Psi operator defined by Psi(X,Y) = sum_j X_ij Y_ij

    Args:
        X (nd array): a N x M matrix
        Y (nd array): a N x M matrix

    Returns:
        nd array: N dimensional vector
    """
    return np.sum(X * Y, axis=1)


def vec(X):
    """vectorization of a matrix X

    Args:
        X (nd array): a (N x M) matrix

    Returns:
        nd array: NM dimensional vector
    """
    return X.flatten()


def mat(xv, matrix_shape):
    """matrixization of a vector xv

    Args:
        xv (nd array): a vector 
        matrix_shape (shape): matrix shape

    Returns:
        nd array: a (NxM) matrix
    """
    return xv.reshape(matrix_shape)

