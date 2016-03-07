
import numpy as np
from functools import partial

try:
    from radio_beam import Beam
    _radio_beam_flag = True
except ImportError:
    Warning("radio_beam is not installed.")
    _radio_beam_flag = False


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


def beam_struct(beam, scale, pixscale, return_beam=False):
    '''
    Return a beam structure.
    '''
    if not _radio_beam_flag:
        raise ImportError("radio_beam must be installed to return a beam"
                          " structure.")

    if scale == 1:
        scale_beam = beam
    else:
        scale_beam = Beam(major=scale*beam.major,
                          minor=scale*beam.minor,
                          pa=beam.pa)

    struct = scale_beam.as_tophat_kernel(pixscale).array
    struct = (struct > 0).astype(int)

    if return_beam:
        return struct, scale_beam

    return struct


def sig_clip(array, nsig=6, tol=0.01, max_iters=500,
             return_clipped=False):
    '''
    Sigma clipping based on the getsources method.
    '''
    nsig = float(nsig)
    mask = np.isfinite(array)
    std = np.nanstd(array)
    thresh = nsig * std

    iters = 0
    while True:
        good_pix = np.abs(array*mask) <= thresh
        new_thresh = nsig * np.nanstd(array[good_pix])
        diff = np.abs(new_thresh - thresh) / thresh
        thresh = new_thresh

        if diff <= tol:
            break
        elif iters == max_iters:
            raise ValueError("Did not converge")
        else:
            iters += 1
            continue

    sigma = thresh / nsig
    if not return_clipped:
        return sigma

    output = array.copy()
    output[output < thresh] = np.NaN

    return sigma, output
