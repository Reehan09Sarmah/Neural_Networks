import numpy as np


def my_sigmoid(z):
    return 1/(1 + np.exp(-z))


def my_dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for i in range(units):
        w = W[:, i]
        z = np.dot(w, a_in) + b[i]
        a_out[i] = my_sigmoid(z)

    return a_out


def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x, W1, b1)
    a2 = my_dense(a1, W2, b2)
    return a2


def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m, 1))
    for i in range(m):
        p[i, 0] = my_sequential(X[i], W1, b1, W2, b2)

    return p