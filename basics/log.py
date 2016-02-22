
import numpy as np
from scipy.ndimage import gaussian_laplace
import itertools as itt
import math
from math import sqrt, hypot, log
from numpy import arccos
from skimage.util import img_as_float
from skimage.feature import peak_local_max
# from .._shared.utils import assert_nD

'''
Copyright (C) 2011, the scikit-image team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
 3. Neither the name of skimage nor the names of its contributors may be
    used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

# This basic blob detection algorithm is based on:
# http://www.cs.utah.edu/~jfishbau/advimproc/project1/ (04.04.2013)
# Theory behind: http://en.wikipedia.org/wiki/Blob_detection (04.04.2013)


def _blob_overlap(blob1, blob2):
    """Finds the overlapping area fraction between two blobs.

    Returns a float representing fraction of overlapped area.

    Parameters
    ----------
    blob1 : sequence
        A sequence of ``(y,x,sigma)``, where ``x,y`` are coordinates of blob
        and sigma is the standard deviation of the Gaussian kernel which
        detected the blob.
    blob2 : sequence
        A sequence of ``(y,x,sigma)``, where ``x,y`` are coordinates of blob
        and sigma is the standard deviation of the Gaussian kernel which
        detected the blob.

    Returns
    -------
    f : float
        Fraction of overlapped area.
    """
    root2 = sqrt(2)

    # extent of the blob is given by sqrt(2)*scale
    r1 = blob1[2]  # * root2
    r2 = blob2[2]  # * root2

    d = hypot(blob1[0] - blob2[0], blob1[1] - blob2[1])

    if d > r1 + r2:
        return 0

    # one blob is inside the other, the smaller blob must die
    if d <= abs(r1 - r2):
        return 1

    ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    ratio1 = np.clip(ratio1, -1, 1)
    acos1 = arccos(ratio1)

    ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    ratio2 = np.clip(ratio2, -1, 1)
    acos2 = arccos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = r1 ** 2 * acos1 + r2 ** 2 * acos2 - 0.5 * sqrt(abs(a * b * c * d))

    return area / (math.pi * (min(r1, r2) ** 2))


def _prune_blobs(blobs_array, overlap):
    """Eliminated blobs with area overlap.

    Parameters
    ----------
    blobs_array : ndarray
        A 2d array with each row representing 3 values, ``(y,x,sigma)``
        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel which detected the blob.
    overlap : float
        A value between 0 and 1. If the fraction of area overlapping for 2
        blobs is greater than `overlap` the smaller blob is eliminated.

    Returns
    -------
    A : ndarray
        `array` with overlapping blobs removed.
    """

    # iterating again might eliminate more blobs, but one iteration suffices
    # for most cases
    for blob1, blob2 in itt.combinations(blobs_array, 2):
        if _blob_overlap(blob1, blob2) > overlap:
            if blob1[2] > blob2[2]:
                blob2[2] = -1
            else:
                blob1[2] = -1

    # return blobs_array[blobs_array[:, 2] > 0]
    return np.array([b for b in blobs_array if b[2] > 0])


def blob_log(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=.2,
             overlap=.5, log_scale=False, sigma_ratio=2.):
    """Finds blobs in the given grayscale image.

    Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    Parameters
    ----------
    image : ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : float, optional
        The minimum standard deviation for Gaussian Kernel. Keep this low to
        detect smaller blobs.
    max_sigma : float, optional
        The maximum standard deviation for Gaussian Kernel. Keep this high to
        detect larger blobs.
    num_sigma : int, optional
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.

    Returns
    -------
    A : (n, 3) ndarray
        A 2d array with each row representing 3 values, ``(y,x,sigma)``
        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel which detected the blob.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian

    Examples
    --------
    >>> from skimage import data, feature, exposure
    >>> img = data.coins()
    >>> img = exposure.equalize_hist(img)  # improves detection
    >>> feature.blob_log(img, threshold = .3)
    array([[ 113.        ,  323.        ,    1.        ],
           [ 121.        ,  272.        ,   17.33333333],
           [ 124.        ,  336.        ,   11.88888889],
           [ 126.        ,   46.        ,   11.88888889],
           [ 126.        ,  208.        ,   11.88888889],
           [ 127.        ,  102.        ,   11.88888889],
           [ 128.        ,  154.        ,   11.88888889],
           [ 185.        ,  344.        ,   17.33333333],
           [ 194.        ,  213.        ,   17.33333333],
           [ 194.        ,  276.        ,   17.33333333],
           [ 197.        ,   44.        ,   11.88888889],
           [ 198.        ,  103.        ,   11.88888889],
           [ 198.        ,  155.        ,   11.88888889],
           [ 260.        ,  174.        ,   17.33333333],
           [ 263.        ,  244.        ,   17.33333333],
           [ 263.        ,  302.        ,   17.33333333],
           [ 266.        ,  115.        ,   11.88888889]])

    Notes
    -----
    The radius of each blob is approximately :math:`\sqrt{2}sigma`.
    """

    # assert_nD(image, 2)

    image = img_as_float(image)

    # if log_scale:
    #     start, stop = log(min_sigma, 10), log(max_sigma, 10)
    #     sigma_list = np.logspace(start, stop, num_sigma)
    # else:
    #     sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    k = int(log(float(max_sigma) / min_sigma, sigma_ratio)) + 1

    # a geometric progression of standard deviations for gaussian kernels
    sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                           for i in range(k)])

    # computing gaussian laplace
    # s**2 provides scale invariance
    gl_images = [gaussian_laplace(image, s) * s ** 2 for s in sigma_list]
    image_cube = np.dstack(gl_images)

    local_maxima = peak_local_max(image_cube, threshold_abs=threshold,
                                  footprint=np.ones((3, 3, 3)),
                                  threshold_rel=0.0,
                                  exclude_border=False)

    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)
    # Convert the last index to its corresponding scale value
    lm[:, 2] = sigma_list[local_maxima[:, 2]] * np.sqrt(2)
    local_maxima = lm
    # return local_maxima, image_cube
    return _prune_blobs(local_maxima, overlap), image_cube
