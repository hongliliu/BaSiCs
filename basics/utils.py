
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

    # Check if a quantity has been passed
    if hasattr(array, "unit"):
        unit = array.unit
        array = array.value.copy()
    else:
        unit = 1.

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

    sigma *= unit
    output *= unit

    return sigma, output

'''
Scipy license:

Copyright (c) 2001, 2002 Enthought, Inc.
All rights reserved.

Copyright (c) 2003-2012 SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of Enthought nor the names of the SciPy Developers
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
'''


def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)

    return a, outaxis


def mode(a, axis=None):
    """
    Returns an array of the modal (most common) value in the passed array.
    If there is more than one such value, the maximum is returned.

    The scipy.stats.mode function returns the smallest value when there are
    multiple modes. This version returns the largest.

    Parameters
    ----------
    a : array_like
        n-dimensional array of which to find mode(s).
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.

    Returns
    -------
    mode : float
        The mode of the array. If there are more than one, the largest is
        returned.

    Examples
    --------
    >>> a = np.array([[6, 8, 3, 0],
    ...               [3, 2, 1, 7],
    ...               [8, 1, 8, 4],
    ...               [5, 3, 0, 5],
    ...               [4, 7, 5, 9]])
    >>> from scipy import stats
    >>> stats.mode(a)
    (array([[3, 1, 0, 0]]), array([[1, 1, 1, 1]]))
    To get mode of whole array, specify ``axis=None``:
    >>> stats.mode(a, axis=None)
    (array([3]), array([3]))

    """
    a, axis = _chk_asarray(a, axis)
    if a.size == 0:
        return np.array([]), np.array([])

    scores = np.unique(np.ravel(a))[::-1]  # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape, dtype=a.dtype)
    oldcounts = np.zeros(testshape, dtype=int)
    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis), axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent


def consec_split(data, stepsize=1):
    '''
    Split an array into consecutive sequences.
    http://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
    '''
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def find_nearest(array, value):
    '''
    http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    '''
    idx = (np.abs(array-value)).argmin()
    return array[idx]
