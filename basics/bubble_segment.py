
import numpy as np
import warnings
from astropy.nddata.utils import extract_array, add_array
from astropy.utils.console import ProgressBar
import astropy.units as u
import skimage.morphology as mo
import skimage.measure as me
from skimage.filters import threshold_adaptive
from skimage.segmentation import find_boundaries, clear_border
from skimage.restoration import denoise_bilateral
from skimage.exposure import rescale_intensity
from skimage.feature import peak_local_max
import scipy.ndimage as nd

try:
    import cv2
    CV2_FLAG = True
except ImportError:
    import scipy.ndimage as nd
    warnings.warn("Cannot import cv2. Computing with scipy.ndimage")
    CV2_FLAG = False

from radio_beam import Beam
from spectral_cube.lower_dimensional_structures import LowerDimensionalObject

from basics.utils import arctan_transform

eight_conn = np.ones((3, 3))


class BubbleSegment(object):
    """
    Image segmentation for bubbles in a 2D image.
    """
    def __init__(self, array, scales=[], atan_transform=True, threshold=None,
                 mask=None, cut_to_box=False, pad_size=0, structure="beam",
                 beam=None, wcs=None):

        if isinstance(array, LowerDimensionalObject):
            self.array = array.value
            self.wcs = array.wcs

            if 'beam' in array.meta:
                self.beam = array.meta['beam']
            elif beam is not None:
                self.beam = beam
            else:
                raise KeyError("No 'beam' in metadata. Must manually specify "
                               "the beam with the beam keyword.")

        elif isinstance(array, np.ndarray):
            self.array = array

            if beam is not None:
                self.beam = beam
            else:
                raise KeyError("Must specify the beam with the beam keyword.")

            if wcs is not None:
                self.wcs = wcs
            else:
                raise KeyError("Must specify the wcs with the wcs keyword.")

        self._threshold = threshold
        self.mask = mask
        self.pad_size = pad_size
        self.scales = scales

        self._atan_flag = False

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask_array):
        if mask_array is None:
            self._mask = np.ones_like(mask_array).astype(bool)
        else:
            if mask_array.shape != self.array.shape:
                raise TypeError("mask must match the shape of the given "
                                "array.")
            self._mask = mask_array

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, input_array):
        if input_array.ndim != 2:
            raise TypeError("Given array must be 2D.")

        self._array = input_array

    @property
    def pad_size(self):
        return self._pad_size

    @pad_size.setter
    def pad_size(self, value):
        if value < 0:
            raise ValueError("Pad size must be >=0")
        self._pad_size = value

    def apply_atan_transform(self, threshold=None):

        if self._atan_flag:
            warnings.warn("arctan transform already applied to the data.")
            return

        if threshold is not None:
            self._threshold = threshold

        self._array = arctan_transform(self.array, self._threshold)
        self._atan_flag = True

    def cut_to_bounding_box(self, pad_size=0):
        '''
        Reduce the array down to the minimum size based on the mask.
        Optionally add padding to the reduced size.
        '''

        if pad_size != self.pad_size:
            self.pad_size = pad_size

        mask_pixels = np.where(self.mask)

        yedges = [mask_pixels[0].min(), mask_pixels[0].max()]
        xedges = [mask_pixels[1].min(), mask_pixels[1].max()]

        self._center_coords = ((yedges[1] + 1 + yedges[0])/2,
                               (xedges[1] + 1 + xedges[0])/2)

        cut_shape = (yedges[1]+1-yedges[0], xedges[1]+1-xedges[0])

        cut_arr = extract_array(self.array, cut_shape, self.center_coords)
        cut_mask = extract_array(self.mask, cut_shape, self.center_coords)

        if self.pad_size > 0:
            # Pads edges with zeros
            self.array = np.pad(cut_arr, self.pad_size, mode='constant')
            self.mask = np.pad(cut_mask, self.pad_size, mode='constant')
        else:
            self.array = cut_arr
            self.mask = cut_mask

    @property
    def center_coords(self):
        return self._center_coords

    def insert_in_shape(self, shape):
        '''
        Insert the cut down mask into the given shape.
        '''

        if self.bubble_mask.shape == shape:
            return self.bubble_mask
        else:
            full_size = np.zeros(shape)
            return add_array(full_size, self.bubble_mask, self.center_coords)

    @property
    def bubble_mask(self):
        return self._bubble_mask

    def apply_bilateral_filter(self, **kwargs):
        '''
        Apply a denoising bilateral filter to the array.
        '''

        # Have to rescale the image to be positive. Pick 0-1
        self.array = rescale_intensity(self.array, out_range=(0.0, 1.0))

        self.array = denoise_bilateral(self.array, **kwargs)

    def multiscale_bubblefind(self, scales=None, emission_reject=True):
        '''
        Run find_bubbles on the specified scales.
        '''

        if scales is not None:
            self.scales = scales

        self._bubble_mask = \
            np.zeros((len(self.scales), ) + self.array.shape, dtype=bool)

        pixscale = np.abs(self.wcs.pixel_scale_matrix[0, 0])
        min_distance = np.ceil(self.beam.minor.value/(2*pixscale))

        for i, scale in enumerate(ProgressBar(self.scales)):
            if not emission_reject:
                mask = find_bubbles(self.array, scale,
                                    self.beam, self.wcs)
            else:
                holes = find_bubbles(self.array, scale, self.beam, self.wcs)
                emission = find_emission(self.array, scale, self.beam,
                                         self.wcs)

                mask = holes * np.logical_not(emission)

            self._bubble_mask = remove_spurs(mask, min_distance=min_distance)

        self._bubble_mask = region_rejection(self._bubble_mask, self.array)


def find_bubbles(array, scale, beam, wcs, min_scale=2):

    # In deg/pixel
    # pixscale = get_pixel_scales(wcs) * u.deg
    pixscale = np.abs(wcs.pixel_scale_matrix[0, 0])

    struct, scale_beam = beam_struct(beam, scale, pixscale,
                                     return_beam=True)
    struct_orig, beam_orig = \
        beam_struct(beam, min_scale, pixscale, return_beam=True)

    # Black tophat
    if CV2_FLAG:
        array = array.astype("float64")
        struct = struct.astype("uint8")
        bth = cv2.morphologyEx(array, cv2.MORPH_BLACKHAT, struct)
    else:
        bth = nd.black_tophat(array, structure=struct)

    # Adaptive threshold
    adapt = \
        threshold_adaptive(bth,
                           int(np.ceil((scale_beam.major/pixscale).value)),
                           param=np.ceil(scale_beam.major.value/pixscale)/2,
                           offset=-np.percentile(bth, 5))

    # Open/close to clean things up
    if CV2_FLAG:
        struct_orig = struct_orig.astype("uint8")
        opened = cv2.morphologyEx(adapt.astype("uint8"), cv2.MORPH_OPEN,
                                  struct_orig)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, struct_orig)
    else:
        opened = nd.binary_opening(adapt, structure=struct_orig)
        closed = nd.binary_closing(opened, structure=struct_orig)

    # # Remove elements smaller than the original beam.
    beam_pixels = np.floor(beam.sr.to(u.deg**2)/pixscale**2).astype(int).value
    cleaned = mo.remove_small_objects(closed, min_size=beam_pixels,
                                      connectivity=2)

    return cleaned


def find_emission(array, scale, beam, wcs, min_scale=2):

    # In deg/pixel
    # pixscale = get_pixel_scales(wcs) * u.deg
    pixscale = np.abs(wcs.pixel_scale_matrix[0, 0])

    struct, scale_beam = beam_struct(beam, scale, pixscale,
                                     return_beam=True)
    struct_orig, beam_orig = \
        beam_struct(beam, min_scale, pixscale, return_beam=True)

    # Black tophat
    if CV2_FLAG:
        array = array.astype("float64")
        struct = struct.astype("uint8")
        wth = cv2.morphologyEx(array, cv2.MORPH_TOPHAT, struct)
    else:
        wth = nd.white_tophat(array, structure=struct)

    # Adaptive threshold
    adapt = \
        threshold_adaptive(wth,
                           int(np.ceil((scale_beam.major/pixscale).value)),
                           param=np.ceil(scale_beam.major.value/pixscale)/2,
                           offset=-np.percentile(wth, 5))

    # Open/close to clean things up
    if CV2_FLAG:
        struct_orig = struct_orig.astype("uint8")
        opened = cv2.morphologyEx(adapt.astype("uint8"), cv2.MORPH_OPEN,
                                  struct_orig)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, struct_orig)
    else:
        opened = nd.binary_opening(adapt, structure=struct_orig)
        closed = nd.binary_closing(opened, structure=struct_orig)

    # # Remove elements smaller than the original beam.
    beam_pixels = np.floor(beam.sr.to(u.deg**2)/pixscale**2).astype(int).value
    cleaned = mo.remove_small_objects(closed, min_size=beam_pixels,
                                      connectivity=2)

    return cleaned


def beam_struct(beam, scale, pixscale, return_beam=False):
    '''
    Return a beam structure.
    '''

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


def region_rejection(bubble_mask_cube, array, grad_thresh=1, frac_thresh=0.05,
                     border_clear=True):
    '''
    2D bubble candidate rejection.
    '''

    spec_shape = bubble_mask_cube.shape[0]

    dy, dx = np.gradient(array, 9)
    magnitude = np.sqrt(dy**2+dx**2)
    orientation = np.arctan2(dy, dx)

    grad_thresh = np.mean(magnitude) + grad_thresh * np.std(magnitude)

    magnitude_mask = nd.median_filter(magnitude > grad_thresh,
                                      footprint=eight_conn)

    for i in range(spec_shape-1):

        # Remove any pixels in this level if the next level doesn't contain it
        # in_smaller = \
        #     np.logical_xor(bubble_mask_cube[i], bubble_mask_cube[i+1]) & \
        #     bubble_mask_cube[i]

        # bubble_mask_cube[i][in_smaller] = False

        if border_clear:
            bubble_mask_cube[i] = clear_border(bubble_mask_cube[i])

        labels, n = me.label(bubble_mask_cube[i], neighbors=8, connectivity=2,
                             return_num=True)

        boundaries = find_boundaries(labels, connectivity=2)

        perimeters = nd.sum(boundaries, labels, range(1, n+1))
        perimeter_masked = nd.sum(boundaries*magnitude_mask, labels,
                                  range(1, n+1))

        masked_fractions = \
            [mask/float(perim) for mask, perim in
             zip(perimeter_masked, perimeters)]

        remove_mask = np.zeros_like(array, dtype=bool)
        for j, frac in enumerate(masked_fractions):
            if frac < frac_thresh:
                bubble_mask_cube[i][np.where(labels == j+1)] = 0
                remove_mask[np.where(labels == j+1)] = 1

        # import matplotlib.pyplot as p
        # p.subplot(121)
        # p.imshow(magnitude_mask, origin='lower')
        # try:
        #     p.contour(remove_mask, colors='r')
        # except:
        #     pass
        # p.subplot(122)
        # p.imshow(array, origin='lower')
        # try:
        #     p.contour(remove_mask, colors='r')
        # except:
        #     pass
        # p.show()
        # raw_input("?")
        # p.clf()

    return bubble_mask_cube


def auto_watershed(mask, array, min_distance=9):
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

    return reconst > 0
