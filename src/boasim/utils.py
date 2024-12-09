import numpy as np

# source: https://stackoverflow.com/a/15196628
def polyfit_with_fixed_points(n, x, y, xf, yf) :
    """
    Fit a polynomial through a given set of points. Some of the points are forced (interpolation).
    Some of them are fitted by regression.

    :param n:   polynomial degree
    :param x:   sequence of x values to fit (regression)
    :param y:   sequence of y values to fit (regression)
    :param xf:  sequence of fixed x values to fit exactly
    :param yf:  sequence of fixed y values to fit exactly
    """
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x**np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf**np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:n + 1]