
import numpy as np
from functools import partial


def arctan_transform(array, thresh):
    return np.arctan(array/thresh)


def dist_uppertri(cond_arr, shape):
    '''
    Convert a condensed distance matrix into the upper triangle. This is what
    squareform in scipy.spatial.distance does, without filling the lower
    triangle.

    From: http://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist

    Parameters
    ----------
    cond_arr : np.ndarray
        Condensed distance matrix from pdist.
    shape : int
        Number of entries in the data used to calculate the distance matrix.

    Returns
    -------
    dist_arr : np.ndarray
        Upper triangular distance matrix.
    '''
    dist_arr = np.zeros((shape, ) * 2, dtype=cond_arr.dtype)

    def unrav_ind(i, j, n):
        return n*j - j*(j+1)/2 + i - 1 - j

    arr_ind = partial(unrav_ind, n=shape)

    for i in xrange(shape):
        for j in xrange(i):
            dist_arr[i, j] = cond_arr[arr_ind(i, j)]

    return dist_arr
