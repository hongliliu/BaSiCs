
from spectral_cube import SpectralCube
from astropy import units as u
import numpy as np
from astropy.utils.console import ProgressBar
from astropy.convolution import MexicanHat2DKernel, convolve_fft
import os
from itertools import izip

# from basics.iterative_watershed import iterative_watershed
execfile("basics/iterative_watershed.py")
from basics.bubble_segment import BubbleSegment


data_path = "/media/eric/Data_3/VLA/IC1613/"
# data_path = "/Users/eric/Data/"

cube = SpectralCube.read(os.path.join(data_path, "IC1613_NA_ICL001.fits"))

# Remove empty channels
cube = cube[38:63, 500:1500, 500:1500]

# Test on some of the central channels
test_cube = np.empty((25, 1000, 1000), dtype='uint8')
peaks = {}
region_props = {}

for i, j in enumerate(range(25)):
    bub = BubbleSegment(cube[j])
    bub.apply_bilateral_filter()
    bub.multiscale_bubblefind()
    test_cube[i] = bub.bubble_mask[1]
    peaks[i] = bub.peaks_dict[8.0]
    region_props[i] = bub.region_props

num_channels = (test_cube > 0).sum(0)
clean_cube = (test_cube > 0).astype(int)

forward = np.roll(clean_cube, 1, axis=0) - clean_cube
backward = np.roll(clean_cube, -1, axis=0) - clean_cube

extents = np.logical_or(forward == -1, backward == -1)

has_one = np.any(extents, axis=0)

# for y, x in izip(*np.where(has_one)):
#     chan_diff = np.diff(np.where(extents[:, y, x]))
#     if not np.any(chan_diff):
#         # Remove that object
#         chan = np.where(extents[:, y, x])[0][0]
#         lab = test_cube[chan, y, x]
#         clean_cube[chan, np.where(test_cube[chan] == lab)] = 0
