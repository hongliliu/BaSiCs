
from spectral_cube import SpectralCube
import os
import numpy as np
from basics.bubble_segment2D import BubbleFinder2D
from basics.utils import sig_clip
# execfile("basics/bubble_segment2D.py")

# data_path = "/media/eric/Data_3/LITTLE_THINGS/IC1613/"
data_path = "/media/eric/Data_3/LITTLE_THINGS/IC10/"
# data_path = "/media/eric/MyRAID/M33/14B-088/HI/full_imaging/"
# data_path = "/home/ekoch/Data/IC1613/"
# data_path = "/home/ekoch/Data/IC10/"
# data_path = "/Users/eric/Data/"
# data_path = "/media/eric/Data_3/VLA/THINGS/HO_II/"

# cube = SpectralCube.read(os.path.join(data_path, "IC1613_NA_ICL001.fits"))
cube = SpectralCube.read(os.path.join(data_path, "IC10_NA_ICL001.fits"))
# cube = SpectralCube.read(os.path.join(data_path, "M33_14B-088_HI.clean.image.pbcov_gt_0.3_masked.fits"))
# cube = SpectralCube.read(os.path.join(data_path, "HO_II_NA_CUBE_THINGS.FITS"))

# Remove empty channels
# cube = cube[:, 500:1500, 500:1500]
# cube = cube[:, 500:1500, 500:1500]

scales = 3. * np.arange(1, 15, np.sqrt(2))

# Find sigma in an empty channel
# sigma = sig_clip(cube[-1].value, nsig=10)
# bubble_15 = BubbleFinder2D(cube[200], channel=15, sigma=sigma, auto_cut=True,
#                            scales=scales)
sigma = sig_clip(cube[0].value, nsig=10)
bubble_15 = BubbleFinder2D(cube[53], channel=15, sigma=sigma, auto_cut=True)
bubble_15.multiscale_bubblefind(overlap_frac=0.5, edge_find=True,
                                nsig=1.5, verbose=False, min_in_mask=0.75)

bubble_15.visualize_regions(show=True, edges=True, region_col='b',
                            edge_col='g')
