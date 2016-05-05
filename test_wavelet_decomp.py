
from spectral_cube import SpectralCube
import os

execfile("basics/bubble_segment2D.py")

# data_path = "/media/eric/Data_3/LITTLE_THINGS/IC1613/"
# data_path = "/media/eric/Data_3/LITTLE_THINGS/IC10/"
# data_path = "/media/eric/MyRAID/M33/14B-088/HI/full_imaging/"
data_path = "/home/ekoch/Data/IC1613/"
# data_path = "/home/ekoch/Data/IC10/"
# data_path = "/Users/eric/Data/"

cube = SpectralCube.read(os.path.join(data_path, "IC1613_NA_ICL001.fits"))
# cube = SpectralCube.read(os.path.join(data_path, "IC10_NA_ICL001.fits"))
# cube = SpectralCube.read(os.path.join(data_path, "M33_14B-088_HI.clean.image.pbcov_gt_0.3_masked.fits"), mode='denywrite')

# Remove empty channels
# cube = cube[:, 500:1500, 500:1500]
cube = cube[38:63, 500:1500, 500:1500]

# Find sigma in an empty channel
sigma = sig_clip(cube[0].value, nsig=10)
bubble_15 = BubbleFinder2D(cube[15], channel=15, sigma=sigma)
bubble_15.multiscale_bubblefind(overlap_frac=0.5, edge_find=True,
                                nsig=1.5, verbose=True)

bubble_15.visualize_regions(show=True, edges=False, region_col='b',
                            edge_col='g')
