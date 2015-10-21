
from radio_beam import Beam
from signal_id.utils import get_pixel_scales
import skimage.morphology as mo
from astropy.convolution import convolve_fft
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

            struct = scale_beam.as_tophat_kernel(pixscale).array
            # Remove empty space along the edges
            yposn, xposn = np.where(struct > 0)
            struct = \
                struct[np.min(yposn):np.max(yposn), np.min(xposn):np.max(xposn)]

            bths[i, :, :] = nd.black_tophat(array, structure=struct)

    return bths
