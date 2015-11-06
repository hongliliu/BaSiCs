
from radio_beam import Beam
from signal_id.utils import get_pixel_scales
import skimage.morphology as mo
from skimage.filters import threshold_adaptive
from astropy.convolution import convolve_fft
import astropy.units as u
import scipy.ndimage as nd
import matplotlib.pyplot as p
import numpy as np


def prog_BTH(array, scales=np.linspace(5, 200, 20), structures=None,
             wcs=None, beam=None):

    bths = np.empty((len(scales), ) + array.shape)

    if beam is None:
        if structures is not None:
            for i, struct in enumerate(structures):
                bths[i, :, :] = nd.black_tophat(array, structure=struct)
        else:
            for i, scale in enumerate(scales):
                bths[i, :, :] = nd.black_tophat(array, size=int(scale))

    else:
        if wcs is None:
            raise TypeError("Must specify wcs when providing a beam.")

        pixscale = get_pixel_scales(wcs)

        for i, scale in enumerate(scales):

            print scale

            if scale == 1:
                scale_beam = beam
            if scale > 1:
                scale_beam = Beam(major=scale*beam.major,
                                  minor=scale*beam.minor,
                                  pa=beam.pa)

            struct = scale_beam.as_tophat_kernel(pixscale).array > 0.0
            struct = struct.astype(int)
            # Remove empty space along the edges
            yposn, xposn = np.where(struct > 0)
            struct = \
                struct[np.min(yposn)-1:np.max(yposn)+2,
                       np.min(xposn)-1:np.max(xposn)+2]

            bths[i, :, :] = nd.black_tophat(array, structure=struct)

    return bths


def find_bubbles(array, scale, beam, wcs):

    # In deg/pixel
    # pixscale = get_pixel_scales(wcs) * u.deg
    pixscale = np.abs(wcs.pixel_scale_matrix[0, 0])

    if scale == 1:
        scale_beam = beam
    else:
        scale_beam = Beam(major=scale*beam.major,
                          minor=scale*beam.minor,
                          pa=beam.pa)

    struct = scale_beam.as_tophat_kernel(pixscale).array > 0.0
    struct = struct.astype(int)
    # Remove empty space along the edges
    yposn, xposn = np.where(struct > 0)
    struct = \
        struct[np.min(yposn)-1:np.max(yposn)+2,
               np.min(xposn)-1:np.max(xposn)+2]

    struct_orig = beam.as_tophat_kernel(pixscale).array > 0.0
    struct_orig = struct_orig.astype(int)
    # Remove empty space along the edges
    yposn, xposn = np.where(struct_orig > 0)
    struct_orig = \
        struct_orig[np.min(yposn)-1:np.max(yposn)+2,
                    np.min(xposn)-1:np.max(xposn)+2]

    # Black tophat
    bth = nd.black_tophat(array, structure=struct)

    # Adaptive threshold
    adapt = \
        threshold_adaptive(bth,
                           int(np.floor((scale_beam.major/pixscale).value)),
                           param=np.floor(scale_beam.major.value/pixscale)/2)

    # Open/close to clean things up
    opened = nd.binary_opening(adapt, structure=struct_orig)
    closed = nd.binary_closing(opened, structure=struct_orig)

    # Remove elements smaller than the original beam.
    beam_pixels = np.floor(beam.sr.to(u.deg**2)/pixscale**2).astype(int).value
    cleaned = mo.remove_small_objects(closed, min_size=beam_pixels,
                                      connectivity=2)

    # p.imshow(struct)
    # raw_input("?")
    # p.imshow(struct_orig)
    # raw_input("?")
    # p.imshow(bth)
    # raw_input("?")
    # p.imshow(adapt)
    # raw_input("?")
    # p.imshow(opened)
    # raw_input("?")
    # p.imshow(closed)
    # raw_input("?")
    # p.imshow(cleaned)
    # raw_input("?")

    return cleaned
