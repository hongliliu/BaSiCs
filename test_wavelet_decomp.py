
import numpy as np
import matplotlib.pyplot as p
from matplotlib.patches import Ellipse
from spectral_cube import SpectralCube
import os
import scipy.ndimage as nd

# import cv2

execfile("basics/bubble_segment2D.py")
execfile("basics/iterative_watershed.py")


data_path = "/media/eric/Data_3/VLA/IC1613/"
# data_path = "/Users/eric/Data/"

cube = SpectralCube.read(os.path.join(data_path, "IC1613_NA_ICL001.fits"))

# Remove empty channels
cube = cube[38:65, 500:1500, 500:1500]

# Find sigma in an empty channel
sigma = sig_clip(cube[0].value, nsig=10)

bubble_15 = BubbleFinder2D(cube[15])
# bubble_15.apply_bilateral_filter()
bubble_15.multiscale_bubblefind(sigma=sigma, overlap_frac=0.7)
# bubble_15.region_rejection(value_thresh=3*sigma)
# bubble_15.apply_atan_transform(np.percentile(bubble_15.array, 90))

cols = ['b', 'g', 'c', 'm', 'r', 'k']

p.ion()

ax = p.subplot(111)

ax.imshow(bubble_15.array, cmap='afmhot')

for blob in bubble_15.region_params:
    y, x, rmaj, rmin, pa = blob
    c = Ellipse((x, y), width=2*rmaj, height=2*rmin, angle=np.rad2deg(pa),
                color='b', fill=False, linewidth=2)
    # c = p.Circle((x, y), r, color='g', fill=False, linewidth=2)
    ax.add_patch(c)
    ax.plot(x, y, 'bD')

p.xlim([0, bubble_15.array.shape[1]])
p.ylim([0, bubble_15.array.shape[0]])

for bub in bubble_15.bubble2D_candidates:
    p.contour(bub.find_shape(cube[15].value, nsig_thresh=0.,
                             return_array='full',
                             value_thresh=3*sigma,
                             linewidth=3), colors='g')
