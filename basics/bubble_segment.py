
import numpy as np
import warnings
from astropy.nddata.utils import extract_array, add_array
from astropy.utils.console import ProgressBar
import astropy.units as u
import skimage.morphology as mo
from skimage.filters import threshold_adaptive

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

    def multiscale_bubblefind(self, scales=None):
        '''
        Run find_bubbles on the specified scales.
        '''

        if scales is not None:
            self.scales = scales

        self._bubble_mask = \
            np.zeros((len(self.scales), ) + self.array.shape, dtype=bool)

        for i, scale in enumerate(ProgressBar(self.scales)):
            self._bubble_mask[i, :, :] = find_bubbles(self.array, scale,
                                                      self.beam, self.wcs)


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
                           param=np.ceil(scale_beam.major.value/pixscale)/2)

    # Open/close to clean things up
    if CV2_FLAG:
        struct_orig = struct_orig.astype("uint8")
        opened = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, struct_orig)
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
