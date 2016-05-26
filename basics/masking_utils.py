
import skimage.morphology as mo
import scipy.ndimage as nd
import warnings
from astropy.modeling.models import Ellipse2D
import numpy as np

try:
    import cv2
    CV2_FLAG = True
except ImportError:
    warnings.warn("Cannot import cv2. Computing with scipy.ndimage")
    CV2_FLAG = False

from utils import eight_conn, ceil_int, floor_int


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


def fraction_in_mask(blob, mask):
    '''
    Find the fraction of a blob within the mask. This is intended to be an
    added check for bad 2D fits.
    '''

    if len(blob) > 3:
        ellipse = Ellipse2D(True, blob[1], blob[0], blob[2], blob[3], blob[4])
    else:
        ellipse = Ellipse2D(True, blob[1], blob[0], blob[2], blob[2], 0.0)

    # Cut the mask to the bounding box
    # yextents, xextents = ellipse.bounding_box

    # Need to round to nearest ints
    # yextents = (max(0, floor_int(yextents[0])),
    #             min(mask.shape[0], ceil_int(yextents[1])))
    # xextents = (max(0, floor_int(xextents[0])),
    #             min(mask.shape[1], ceil_int(xextents[1])))

    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
    # xx, yy = np.mgrid[xextents[0]:xextents[1] + 1,
    #                   yextents[0]:yextents[1] + 1]

    ellip_mask = ellipse(yy, xx).astype(np.bool)

    local_mask = mask
    # local_mask = mask[yextents[0]:yextents[1] + 1,
    #                   xextents[0]:xextents[1] + 1]

    return (ellip_mask * local_mask).sum() / float(ellip_mask.sum())


def fill_nans_with_noise(array, sigma, nsig=2, pad_size=0):
    '''
    Pad the array to avoid edge effects. Fill the NaNs with samples from
    the noise distribution. This does take the correlation of the beam
    out... this is fine for the time being, but adding a quick
    convolution with the beam will make this "proper".
    '''

    all_noise = array <= nsig * sigma
    nans = np.isnan(array)
    samps = np.random.random_integers(0, all_noise.sum() - 1,
                                      size=nans.sum())
    array[nans] = array[all_noise][samps]

    return array
