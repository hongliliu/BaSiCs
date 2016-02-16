
from spectral_cube import SpectralCube
from signal_id.utils import get_pixel_scales
from astropy.io import fits
from astropy.convolution import convolve_fft
import matplotlib.pyplot as p
import numpy as np
import os

from scipy import ndimage as nd
import astropy.units as u

from basics.bubble_finder import BubbleFinder2D

data_path = "/media/eric/Data_3/M33/VLA_Data/AT0206/imaging/"


cube = SpectralCube.read(os.path.join(data_path, "M33_206_b_c_HI.fits"))

clean_mask = fits.getdata("/media/eric/Data_3/M33/Arecibo/AT0206_items/M33_mask.fits")

clean_mask = clean_mask.squeeze()
clean_mask = clean_mask[11:195, 595:3504, 1065:3033]

# cube = cube.with_mask(clean_mask.astype(bool))

bubble_100 = BubbleFinder2D(nd.gaussian_filter(cube[100, :, :].value, 3)*u.Jy, mask=clean_mask[100, :, :])
bubble_100.create_bubble_mask(scales=[3, 5, 8, 10, 15, 20, 30, 50], beam=cube.beam, wcs=cube.wcs)
