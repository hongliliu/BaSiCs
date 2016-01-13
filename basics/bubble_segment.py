
import numpy as np
import warnings
from astropy.nddata.utils import extract_array, add_array
from astropy.utils.console import ProgressBar
from astropy.convolution import convolve_fft, MexicanHat2DKernel
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
    warnings.warn("Cannot import cv2. Computing with scipy.ndimage")
    CV2_FLAG = False

from radio_beam import Beam
from spectral_cube.lower_dimensional_structures import LowerDimensionalObject

from basics.utils import arctan_transform
from basics.iterative_watershed import iterative_watershed

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

        pixscale = np.abs(self.wcs.pixel_scale_matrix[0, 0])
        fwhm_beam_pix = self.beam.major.value / pixscale
        beam_pix = np.ceil(fwhm_beam_pix / np.sqrt(8*np.log(2)))

        # Scales based on 2^n times the major beam radius
        self.scales = beam_pix * 2 ** np.arange(0., 5.1)
        # When using non-gaussian like kernels, adjust the
        # widths to match the FWHM areas
        self.tophat_scales = np.floor(self.scales * np.sqrt(2))

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

    def multiscale_bubblefind(self, scales=None):
        '''
        Run find_bubbles on the specified scales.
        '''

        if scales is not None:
            self.scales = scales

        wave = wavelet_decomp(self.array, self.scales)

        self._bubble_mask = \
            np.zeros((len(self.scales), ) + self.array.shape, dtype=np.uint8)

        self.peaks_dict = dict.fromkeys(self.scales)
        levels = [5, 3, 1.5, 1.5, 1.5, 1.5]

        # Find the stand dev at each scale
        # Normalize each wavelet scale to it
        for i, (arr, scale) in enumerate(zip(wave, self.scales)):
            sigma = sig_clip(arr, nsig=6)
            wave[i] /= sigma
            self._bubble_mask[i], self.peaks_dict[scale] = \
                iterative_watershed(wave[i], self.tophat_scales[i],
                                    end_value=1,
                                    start_value=levels[i],
                                    delta_value=0.25,
                                    mask_below=1)

        # self._bubble_mask = region_rejection(self._bubble_mask, self.array)


def wavelet_decomp(array, scales, kernel=MexicanHat2DKernel):
    '''
    Perform a wavelet decomposition at the given scales.
    Scales correspond to the width of the kernel.
    '''

    # Set nans to the min value
    array[np.isnan(array)] = np.nanmin(array)

    wave = np.zeros((len(scales), ) + array.shape, dtype=np.float)

    for i, scale in enumerate(ProgressBar(scales)):

        kern = -1 * kernel(scale).array

        wave[i] = convolve_fft(array, kern, normalize_kernel=False) * scale**2.

    return wave


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
