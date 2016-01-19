
import skimage.morphology as mo
import skimage.measure as me
from skimage.feature import peak_local_max
import scipy.ndimage as nd
import numpy as np
import warnings

try:
    import cv2
    CV2_FLAG = True
except ImportError:
    warnings.warn("Cannot import cv2. Computing with scipy.ndimage")
    CV2_FLAG = False


def iterative_watershed(array, scale, start_value=5., end_value=3.,
                        delta_value=1., mask_below=2., min_pix=0):
    '''
    Iterative Watershed algorithm.
    '''

    initial_mask = array >= mask_below

    if array.ndim == 3:
        # Define a footprint to use in the peak finding
        footprint = np.tile(mo.disk(scale), (array.shape[0], 1, 1))
        use_footprint = True
        initial_peaks = peak_local_max(array, footprint=footprint,
                                       threshold_abs=start_value,
                                       exclude_border=False)

        initial_markers = np.zeros_like(array, dtype=bool)
        initial_markers[:, initial_peaks[:, 1], initial_peaks[:, 2]] = True

    elif array.ndim == 2:
        initial_peaks = peak_local_max(array, min_distance=scale,
                                       threshold_abs=start_value)
        use_footprint = False

        initial_markers = np.zeros_like(array, dtype=bool)
        initial_markers[initial_peaks[:, 0], initial_peaks[:, 1]] = True
    else:
        raise Exception("Function only implemented for 2D and 3D arrays.")

    initial_markers *= initial_mask

    wshed_input = -array.copy()
    wshed_input[wshed_input > 0] = 0

    labels = mo.watershed(wshed_input, me.label(initial_markers),
                          mask=initial_mask)

    initial_markers *= labels > 0

    # Now decrease the local maxima, trying to subdivide
    # regions that had a peak at the higher level.
    if start_value - end_value < delta_value:
        return labels, np.vstack(np.where(initial_markers)).T

    peak_levels = \
        np.arange(start_value-delta_value,
                  end_value-delta_value, -1*delta_value)

    for value in peak_levels:
        if use_footprint:
            new_peaks = peak_local_max(array, footprint=footprint,
                                       threshold_abs=value,
                                       exclude_border=False)
        else:
            new_peaks = peak_local_max(array, min_distance=scale,
                                       threshold_abs=value)
        markers = initial_markers.copy()
        markers[new_peaks[:, 0], new_peaks[:, 1]] = True

        # Remove markers not in the last watershed
        markers *= labels > 0

        # Search for label regions that now have multiple peaks
        # and re-run the watershed on them
        for lab in range(1, labels.max()+1):
            num_peaks = np.sum(markers*(labels == lab))
            if num_peaks == 1:
                continue
            elif num_peaks == 0:
                raise Exception("No peaks found??")
            else:
                split_marker = me.label(markers*(labels == lab))
                split_label = mo.watershed(wshed_input, split_marker,
                                           mask=labels == lab)
                orig_used = False
                for lab2 in range(1, split_label.max()+1):
                    posns = np.where(split_label == lab2)
                    if array[posns].max() >= start_value:
                        if not orig_used:
                            labels[posns] = lab
                            orig_used = True
                        else:
                            labels[posns] = labels.max() + 1
                    else:
                        labels[posns] = 0

    # Remove small regions
    pix = nd.sum(labels > 0, labels, range(1, labels.max()+1))
    for i in xrange(labels.max()):
        if pix[i] < min_pix:
            labels[labels == i+1] = 0

    markers *= labels > 0

    # Return only the peaks that were used.
    final_peaks = np.vstack(np.where(markers)).T

    return labels, final_peaks


def remove_spurs(mask, min_distance=9):
    '''
    Remove spurious mask features with reconstruction.
    '''

    # Distance transform of the mask
    dist_trans = nd.distance_transform_edt(mask)

    # We don't want to return local maxima within the minimum distance
    # Use reconstruction to remove.
    seed = dist_trans + min_distance
    reconst = mo.reconstruction(seed, dist_trans, method='erosion') - \
        min_distance

    if CV2_FLAG:
        return cv2.morphologyEx((reconst > 0).astype("uint8"),
                                cv2.MORPH_DILATE,
                                mo.disk(min_distance).astype("uint8")).astype(bool)
    else:
        return mo.dilation(reconst > 0, selem=mo.disk(min_distance))


def distance_watershed(mask, array, min_distance=9):
    '''
    Automatically create seeds based on local minima of a distance transform
    and apply the watershed algorithm to find individual regions.
    '''

    # Distance transform of the mask
    dist_trans = nd.distance_transform_edt(mask)

    # We don't want to return local maxima within the minimum distance
    # Use reconstruction to remove.
    seed = dist_trans + min_distance
    reconst = mo.reconstruction(seed, dist_trans, method='erosion') - \
        min_distance

    # Now get local maxima
    coords = peak_local_max(reconst, min_distance=min_distance)

    markers = np.zeros_like(mask)
    markers[coords[:, 0], coords[:, 1]] = True
    markers = me.label(markers, neighbors=8, connectivity=2)

    # Need to reduce the side-by-side ones when there's a flat plateau

    wshed = mo.watershed(array, markers, mask=mask)

    import matplotlib.pyplot as p
    p.imshow(dist_trans)
    p.contour(mask, colors='r')
    p.contour(markers > 0, colors='b')
    raw_input("?")

    return wshed
