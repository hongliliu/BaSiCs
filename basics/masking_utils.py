
import skimage.morphology as mo
import scipy.ndimage as nd
import warnings

try:
    import cv2
    CV2_FLAG = True
except ImportError:
    warnings.warn("Cannot import cv2. Computing with scipy.ndimage")
    CV2_FLAG = False

from utils import eight_conn


def smooth_edges(mask, filter_size, min_pixels):

    no_small = mo.remove_small_holes(mask, min_size=min_pixels,
                                     connectivity=2)

    open_close = \
        nd.binary_closing(nd.binary_opening(no_small, eight_conn), eight_conn)

    medianed = nd.median_filter(open_close, filter_size)

    return mo.remove_small_holes(medianed, min_size=min_pixels,
                                 connectivity=2)


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
