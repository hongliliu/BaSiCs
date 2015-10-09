
from spectral_cube import SpectralCube
from radio_beam import Beam
from signal_id.utils import get_pixel_scales
import skimage.morphology as mo
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve_fft
import scipy.ndimage as nd
import matplotlib.pyplot as p
import numpy as np
import os


def prog_BTH(array, header=None, scales=np.linspace(5, 200, 20)):

    bths = np.empty((len(scales), ) + array.shape)

    if header is None:
        for i, scale in enumerate(scales):
            bths[i, :, :] = nd.black_tophat(array, size=int(scale))

    else:

        mywcs = WCS(header)

        pixscale = get_pixel_scales(mywcs)

        beam = Beam.from_fits_header(header)

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

cube_run = True
mom0_run = False

data_path = "/media/eric/Data_3/M33/VLA_Data/AT0206/imaging/"

if cube_run:

    cube = SpectralCube.read(os.path.join(data_path, "M33_206_b_c_HI.fits"))

    clean_mask = fits.getdata("/media/eric/Data_3/M33/Arecibo/M33_mask.fits")

    clean_mask = clean_mask.squeeze()
    clean_mask = clean_mask[11:195, 595:3504, 1065:3033]

    cube = cube.with_mask(clean_mask.astype(bool))

    beam = cube.beam.as_kernel(get_pixel_scales(cube.wcs)).array

    smoothed = convolve_fft(cube[50:52, :, :].sum(0).value, beam)

    bths_50 = prog_BTH(smoothed, cube.header, scales=range(10, 20))

if mom0_run:

    filepath = os.path.join(data_path, "M33_206_b_c_HI.mom0.fits")

    mom0 = fits.getdata(filepath)

    hdr = fits.getheader(os.path.join(data_path, "M33_206_b_c_HI.fits"))

    bths = prog_BTH(mom0, hdr, scales=range(1, 8))
