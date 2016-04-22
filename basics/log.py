
import numpy as np
from scipy.ndimage import gaussian_laplace
import math
from math import sqrt, hypot, log
from numpy import arccos
from skimage.util import img_as_float
from skimage.feature import peak_local_max
from astropy.modeling.models import Ellipse2D
from skimage.measure import regionprops
from scipy.spatial.distance import pdist, squareform, cdist
from functools import partial

# from .._shared.utils import assert_nD

from basics.utils import dist_uppertri
from basics.utils import wrap_to_pi

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


def _circle_overlap(blob1, blob2, return_corr=False):
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
    # extent of the blob is given by sqrt(2)*scale
    r1 = blob1[2]
    r2 = blob2[2]

    d = hypot(blob1[0] - blob2[0], blob1[1] - blob2[1])

    if d > r1 + r2:
        return 0.0

    # one blob is inside the other, the smaller blob must die
    if d <= abs(r1 - r2):
        if return_corr:
            # Overlap area is the smaller area. This becomes the ratio of the
            # radii!
            if r1 == r2:
                return 1.0
            elif r1 > r2:
                return r2 / float(r1)
            else:
                return r1 / float(r2)

        return 1.0

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

    if return_corr:
        # Aover / sqrt(A1*A2) = Aover = (pi*r1*r2)
        return area / (math.pi * r1 * r2)

    return area / (math.pi * (min(r1, r2) ** 2))


def _prune_blobs(blobs_array, overlap, method='size',
                 min_corr=0.5, return_removal_posns=False,
                 coords=None):
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
        `array` with overlapping blobs removed and merged nearby blobs.
    """

    remove_blobs = []

    # Create a distance matrix to catch all overlap cases.
    cond_arr = pdist(blobs_array[:, :5], metric=partial(overlap_metric))
    dist_arr = dist_uppertri(cond_arr, blobs_array.shape[0])

    overlaps = np.where(dist_arr >= overlap)

    for posn1, posn2 in zip(*overlaps):

        blob1 = blobs_array[posn1]
        blob2 = blobs_array[posn2]

        if method == "shell fraction":
            area1 = np.pi*blob1[2]*blob1[3]
            area2 = np.pi*blob2[2]*blob2[3]

            if area1 >= area2:
                large = blob1
                large_pos = posn1
                small = blob2
                small_pos = posn2
            else:
                large = blob2
                large_pos = posn2
                small = blob1
                small_pos = posn1

            shell_cond = large[5] > small[5] or large[7] <= small[7]
            # Also want to check if the fraction of overlap is above
            # min_corr. This stops much larger regions from being
            # removed when small ones are embedded in their edges.
            overlap_corr = overlap_metric(blob1, blob2,
                                          return_corr=True)
            overlap_cond = overlap_corr >= min_corr

            # If the bigger one is more complete, discard the smaller
            if shell_cond:
                remove_blobs.append(small_pos)
            # Otherwise, only remove the larger one if the smaller overlaps
            # >= min_corr
            else:
                if overlap_cond:
                    remove_blobs.append(large_pos)

        elif method == "shell coords":

            if coords is None:
                raise TypeError("Coordinates must be given for 'shell coords'")

            if len(coords) != len(blobs_array):
                raise TypeError("coords must match the number of blobs in the"
                                " given blob array.")

            area1 = np.pi*blob1[2]*blob1[3]
            area2 = np.pi*blob2[2]*blob2[3]

            if area1 >= area2:
                large = blob1
                large_pos = posn1
                small = blob2
                small_pos = posn2
            else:
                large = blob2
                large_pos = posn2
                small = blob1
                small_pos = posn1

            shell_cond = \
                shell_similarity(coords[posn1], coords[posn2]) > min_corr
            # Discard the smaller one if it is too similar
            if shell_cond:
                remove_blobs.append(small_pos)

        elif method == "size":
            cond = blob1[2] > blob2[2]
            if cond:
                remove_blobs.append(posn2)
            else:
                remove_blobs.append(posn1)

        elif method == "response":
            cond = blob1[5] > blob2[5]
            if cond:
                remove_blobs.append(posn2)
            else:
                remove_blobs.append(posn1)
        else:
            raise TypeError("method must be 'size', 'shell fraction',"
                            " 'response' or 'shell coords'.")
    # Remove duplicates
    remove_blobs = list(set(remove_blobs))

    blobs_array = np.delete(blobs_array, remove_blobs, axis=0)

    if return_removal_posns:
        return blobs_array, remove_blobs

    return blobs_array


def _merge_blobs(blobs_array, min_distance_merge=1.0):
    '''
    Merge blobs together based on a minimum radius overlap.

    Parameters
    ----------
    blobs_array : ndarray
        A 2d array with each row representing 3 values, ``(y,x,sigma)``
        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel which detected the blob.

    min_distance_merge : float
        Number of sigma apart two blobs must be to be eligible for merging.
        Blobs must be the same size to merge.

    '''

    # For merging two overlapping regions into an ellipse
    # Distance between centres equal to the radius
    min_merge_overlap = _min_merge_overlap(min_distance_merge)
    # Distance between centres equal to the radius
    max_merge_overlap = 1.0

    merged_blobs = np.empty((0, 6), dtype=np.float64)

    # Create a distance matrix to catch all overlap cases.
    cond_arr = pdist(blobs_array, metric=partial(overlap_metric))
    dist_arr = dist_uppertri(cond_arr, blobs_array.shape[0])

    overlaps = np.where(np.logical_and(dist_arr > min_merge_overlap,
                                       dist_arr <= max_merge_overlap))

    for posn1, posn2 in zip(*overlaps):

        blob1 = blobs_array[posn1]
        blob2 = blobs_array[posn2]

        if blob1[2] != blob1[3] or blob2[2] != blob2[3]:
            is_ellipse = True
        else:
            is_ellipse = False

        if np.logical_and(blob1[2] == blob2[2], ~is_ellipse):
            # Merge into an ellipse
            merged_blobs = \
                np.vstack([merged_blobs, merge_to_ellipse(blob1, blob2)])
            blob1[2] = -1
            blob1[3] = -1
            blob2[2] = -1
            blob2[3] = -1

    # Remove blobs
    blobs_array = np.array([b for b in blobs_array if b[2] > 0])
    blobs_array = np.vstack([blobs_array, merged_blobs])

    return blobs_array


def blob_log(image, sigma_list=None, scale_choice='linear',
             min_sigma=1, max_sigma=50, num_sigma=10,
             threshold=.2, overlap=.5, sigma_ratio=2.,
             weighting=None, merge_overlap_dist=1.0):
    """Finds blobs in the given grayscale image.

    Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    Parameters
    ----------
    image : ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    sigma_list : np.ndarray, optional
        Provide the list of sigmas to use.
    scale_choice : str, optional
        'log', 'linear' or 'ratio'. Determines how the scales are calculated
        based on the given number, minimum, maximum, or ratio.
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
    weighting : np.ndarray, optional
        Used to weight certain scales differently when selecting local maxima
        in the transform space. For example when searching for regions near
        the beam size, the transform can be down-weighted to avoid spurious
        detections. Must have the same number of elements as the scales.
    merge_overlap_dist : float, optional
        Controls the minimum overlap regions must have to be merged together.
        Defaults to one sigma separation, where sigma is one of the scales
        used in the transform.

    Returns
    -------
    A : (n, 5) ndarray
        A 2d array with each row representing 5 values,
        ``(y, x, semi-major sigma, semi-minor sigma, pa)``
        where ``(y,x)`` are coordinates of the blob, ``semi-major sigma`` and
        ``semi-minor sigma`` are the standard deviations of the elliptical blob,
        and ``pa`` is its position angle.

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

    if sigma_list is None:
        if scale_choice is 'log':
            start, stop = log(min_sigma, 10), log(max_sigma, 10)
            sigma_list = np.logspace(start, stop, num_sigma)
        elif scale_choice is 'linear':
            sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)
        elif scale_choice is 'ratio':
            # k such that min_sigma*(sigma_ratio**k) > max_sigma
            k = int(log(float(max_sigma) / min_sigma, sigma_ratio)) + 1
            sigma_list = np.array([min_sigma * (sigma_ratio ** i)
                                   for i in range(k)])
        else:
            raise ValueError("scale_choice must be 'log', 'linear', or "
                             "'ratio'.")

    if weighting is not None:
        if len(weighting) != len(sigma_list):
            raise IndexError("weighting must have the same number of elements"
                             " as scales (" + str(len(sigma_list)) + ").")
    else:
        weighting = np.ones_like(sigma_list)

    # computing gaussian laplace
    # s**2 provides scale invariance
    # weighting by w changes the relative importance of each transform scale
    gl_images = [gaussian_laplace(image, s) * s ** 2 * w for s, w in
                 zip(sigma_list, weighting)]
    image_cube = np.dstack(gl_images)

    for i, scale in enumerate(sigma_list):
        scale_peaks = peak_local_max(image_cube[:, :, i],
                                     threshold_abs=threshold,
                                     min_distance=np.sqrt(2) * scale,
                                     threshold_rel=0.0,
                                     exclude_border=False)

        new_scale_peaks = np.empty((len(scale_peaks), 6))
        for j, peak in enumerate(scale_peaks):
            new_peak = np.array([peak[0], peak[1], scale, scale, 0.0])
            new_scale_peaks[j] = \
                shape_from_blob_moments(new_peak, image_cube[:, :, i])
            # new_scale_peaks[j, 3:5] *= np.sqrt(2)

        if i == 0:
            local_maxima = new_scale_peaks
        else:
            local_maxima = np.vstack([local_maxima, new_scale_peaks])

    if local_maxima.size == 0:
        return local_maxima

    # Merge regions into ellipses
    local_maxima = _merge_blobs(local_maxima, merge_overlap_dist)

    # Then prune and return them\
    return local_maxima
    # return _prune_blobs(local_maxima, overlap)


def merge_to_ellipse(blob1, blob2):
    '''
    Merge to circular blobs into an elliptical one
    '''

    new_blob = []

    d = hypot(blob1[0] - blob2[0], blob1[1] - blob2[1])

    # Take the average of the centers to find the new center.
    new_blob.append((blob1[0] + blob2[0]) / 2.)
    new_blob.append((blob1[1] + blob2[1]) / 2.)

    # Semi-major axis = r + d/2
    new_blob.append(blob1[2] + d / 2.)

    # Semi-minor axis = r
    new_blob.append(blob1[2])

    # Position angle
    pa = np.arctan((blob1[0] - blob2[0]) / (blob1[1] - blob2[1]))
    # Returns on -pi/2 to pi/2 range. But we want 0 to pi.
    if pa < 0:
        pa += np.pi
    new_blob.append(pa)

    new_blob.append(max(blob1[-1], blob2[-1]))

    return new_blob


def _ellipse_overlap(blob1, blob2, grid_space=0.5, return_corr=False):
    '''
    Ellipse intersection are difficult. But counting common pixels is not!
    This routine creates arrays up-sampled from the original pixel scale to
    better estimate the overlap fraction.
    '''

    ellip1_area = np.pi * blob1[2] * blob1[3]
    ellip2_area = np.pi * blob2[2] * blob2[3]

    if ellip1_area >= ellip2_area:
        large_blob = blob1
        small_blob = blob2
        large_ellip_area = ellip1_area
        small_ellip_area = ellip2_area
    else:
        large_blob = blob2
        small_blob = blob1
        large_ellip_area = ellip2_area
        small_ellip_area = ellip1_area

    bound1 = Ellipse2D(True, 0.0, 0.0, large_blob[2], large_blob[3],
                       large_blob[4]).bounding_box

    bound2 = Ellipse2D(True, small_blob[1]-large_blob[1],
                       small_blob[0]-large_blob[0],
                       small_blob[2], small_blob[3],
                       small_blob[4]).bounding_box

    # If there is no overlap in the bounding boxes, there is no overlap
    if bound1[0][1] <= bound2[0][0] or bound1[1][1] <= bound2[1][0]:
        return 0.0

    # Now check if the smaller one is completely inside the larger
    low_in_large = bound1[0][0] <= bound2[0][0] and \
        bound1[0][1] >= bound2[0][1]
    high_in_large = bound1[1][0] <= bound2[1][0] and \
        bound1[1][1] >= bound2[1][1]

    if low_in_large and high_in_large:
        if return_corr:
            # Aover / sqrt(A1 * A2) = A1 / sqrt(A1 * A2) = sqrt(A1/A2)
            return np.sqrt(small_ellip_area / large_ellip_area)
        return 1.0

    # Only evaluate the overlap between the two.

    ybounds = (max(bound1[0][0], bound2[0][0]),
               min(bound1[0][1], bound2[0][1]))

    xbounds = (max(bound1[1][0], bound2[1][0]),
               min(bound1[1][1], bound2[1][1]))

    yy, xx = \
        np.meshgrid(np.arange(ybounds[0] - grid_space, ybounds[1] + grid_space,
                              grid_space),
                    np.arange(xbounds[0] - grid_space, xbounds[1] + grid_space,
                              grid_space))

    ellip1 = Ellipse2D.evaluate(xx, yy, True, 0.0,
                                0.0, large_blob[2],
                                large_blob[3], large_blob[4])

    ellip2 = Ellipse2D.evaluate(xx, yy, True,
                                small_blob[1]-large_blob[1],
                                small_blob[0]-large_blob[0],
                                small_blob[2], small_blob[3],
                                small_blob[4])

    overlap_area = np.sum(np.logical_and(ellip1, ellip2)) * grid_space ** 2

    if return_corr:
        return overlap_area / np.sqrt(small_ellip_area * large_ellip_area)

    return overlap_area / small_ellip_area


def _min_merge_overlap(min_dist):
    '''
    Parameters
    ----------
    min_dist : float
        Fraction of radius that two circles must be separated
        by to be eligible to be merged.
    '''

    term1 = 2*np.arccos(min_dist/2.) / np.pi
    term2 = np.sqrt((2-min_dist)*(2+min_dist)) / (2*np.pi)

    return term1 - term2


def overlap_metric(ellip1, ellip2, return_corr=False):
    '''
    Calculate the overlap in ellipses, if they are in adjacent channels.
    '''

    if ellip1[2] != ellip1[3] or ellip2[2] != ellip2[3]:
        return _ellipse_overlap(ellip1, ellip2,
                                return_corr=return_corr)

    else:
        blob_overlap = _circle_overlap(ellip1, ellip2,
                                       return_corr=return_corr)
        return blob_overlap


def shell_similarity(coords1, coords2, max_dist=3, verbose=False):
    '''
    Check how similar 2 sets of shell coordinates are. This is based off a
    distance threshold; if two points are within the threshold of each other,
    they are considered a match.
    '''

    # There shouldn't ever be huge number of coordinates for the shells, so
    # the lazy O(n^2) solution is fine for now.

    num1 = len(coords1)
    num2 = len(coords2)

    if num1 >= num2:
        dist = cdist(coords2, coords1)
        ind1 = 1
        ind2 = 0
    else:
        dist = cdist(coords1, coords2)
        ind1 = 0
        ind2 = 1

    # Smaller number of coords is the first dimension
    matches = np.empty((0, 2), dtype=np.int)
    for i, row in enumerate(dist):
        # Find the minimum, excluding points that have already been used.
        candidates = row.copy()
        # Set used ones to NaN
        candidates[matches[:, 1]] = np.NaN
        # Check if all possible matches have already been used
        if np.isnan(candidates).all():
            break
        j = np.nanargmin(candidates)
        if row[j] < max_dist:
            matches = np.append(matches, np.array([[i, j]]), axis=0)

    # Compute the overlap fraction
    if num1 >= num2:
        overlap_frac = matches.shape[0] / float(num2)
    else:
        overlap_frac = matches.shape[0] / float(num1)

    if verbose:
        import matplotlib.pyplot as p
        p.plot(coords1[:, 1], coords1[:, 0], 'bD')
        p.plot(coords2[:, 1], coords2[:, 0], 'gD')
        p.plot(coords1[:, 1][matches[:, ind1]],
               coords1[:, 0][matches[:, ind1]], 'ro')
        p.plot(coords2[:, 1][matches[:, ind2]],
               coords2[:, 0][matches[:, ind2]], 'ro')
        p.show()

    return overlap_frac


def shape_from_blob_moments(blob, response, expand_factor=np.sqrt(2)):
    '''
    Use blob properties to define a region, then correct its shape by
    using the response surface
    '''
    yy, xx = np.mgrid[:response.shape[0], :response.shape[1]]

    mask = Ellipse2D.evaluate(yy, xx, True, blob[0],
                              blob[1], expand_factor * blob[2],
                              expand_factor * blob[3], blob[4]).astype(np.int)

    resp = response.copy()
    # We don't care about the negative values here, so set to 0
    resp[resp < 0.0] = 0.0

    props = regionprops(mask, intensity_image=resp)[0]

    # Now use the moments to refine the shape
    wprops = weighted_props(props)
    y, x, major, minor, pa = wprops

    new_mask = Ellipse2D.evaluate(yy, xx, True, y,
                                  x, major,
                                  minor, pa).astype(bool)

    # Find the maximum value in the response
    max_resp = np.max(response[new_mask])

    # print(blob)
    # print((y, x, major, minor, pa, max_resp))
    # import matplotlib.pyplot as p
    # p.imshow(response, origin='lower', cmap='afmhot')
    # p.contour(mask, colors='r')
    # p.contour(new_mask, colors='g')
    # p.draw()
    # import time; time.sleep(0.1)
    # raw_input("?")
    # p.clf()

    return np.array([y, x, major, minor, pa, max_resp])


def weighted_props(regprops):
    '''
    Return the shape properties based on the weighted moments.
    '''

    wmu = regprops.weighted_moments_central
    # Create the inertia tensor from the moments
    a = wmu[2, 0] / wmu[0, 0]
    b = -wmu[1, 1] / wmu[0, 0]
    c = wmu[0, 2] / wmu[0, 0]
    # inertia_tensor = np.array([[a, b], [b, c]])

    # The eigenvalues give the axes lengths
    l1 = (a + c) / 2 + sqrt(4 * b ** 2 + (a - c) ** 2) / 2
    l2 = (a + c) / 2 - sqrt(4 * b ** 2 + (a - c) ** 2) / 2

    # These are the radii, not the lengths
    major = 2 * np.sqrt(l1)
    minor = 2 * np.sqrt(l2)

    # And finally the orientation
    absb = -b  # need this to be > 0
    if a - c == 0:
        if absb > 0:
            pa = -np.pi / 4.
        else:
            pa = np.pi / 4.
    else:
        pa = - 0.5 * np.arctan2(2 * absb, (a - c))

    pa = wrap_to_pi(pa + 0.5 * np.pi)

    centroid = regprops.weighted_centroid

    return np.array([centroid[0], centroid[1], major, minor, pa])
