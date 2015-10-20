
from spectral_cube import SpectralCube
from signal_id.utils import get_pixel_scales
from astropy.io import fits
from astropy.convolution import convolve_fft
import matplotlib.pyplot as p
import numpy as np
import os


cube_run = False
mom0_run = False

data_path = "/media/eric/Data_3/M33/VLA_Data/AT0206/imaging/"

if cube_run:

    cube = SpectralCube.read(os.path.join(data_path, "M33_206_b_c_HI.fits"))

    clean_mask = fits.getdata("/media/eric/Data_3/M33/Arecibo/M33_mask.fits")

    clean_mask = clean_mask.squeeze()
    clean_mask = clean_mask[11:195, 595:3504, 1065:3033]

    cube = cube.with_mask(clean_mask.astype(bool))

    beam = cube.beam.as_kernel(get_pixel_scales(cube.wcs)).array

    smoothed = convolve_fft(cube[50, :, :].value, beam)

    bths_50 = prog_BTH(smoothed, cube.header, scales=range(10, 20))

if mom0_run:

    filepath = os.path.join(data_path, "M33_206_b_c_HI.mom0.fits")

    mom0 = fits.getdata(filepath)

    hdr = fits.getheader(os.path.join(data_path, "M33_206_b_c_HI.fits"))

    bths = prog_BTH(mom0, hdr, scales=range(1, 8))