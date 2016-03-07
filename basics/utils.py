
import numpy as np
from functools import partial


def arctan_transform(array, thresh):
    return np.arctan(array/thresh)


def dist_uppertri(cond_arr, shape):
    dist_arr = np.zeros((shape, ) * 2, dtype=cond_arr.dtype)

    def unrav_ind(i, j, n):
        return n*j - j*(j+1)/2 + i - 1 - j

    arr_ind = partial(unrav_ind, n=shape)

    for i in xrange(shape):
        for j in xrange(i):
            dist_arr[i, j] = cond_arr[arr_ind(i, j)]

    return dist_arr
