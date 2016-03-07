
from spectral_cube import SpectralCube
from astropy import units as u
import numpy as np
from astropy.utils.console import ProgressBar
from astropy.io import fits
import os
import matplotlib.pyplot as p
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as hier

from basics.bubble_segment import BubbleFinder2D
from basics.utils import sig_clip
from basics.log import overlap_metric

# from basics.iterative_watershed import iterative_watershed
execfile("basics/iterative_watershed.py")


data_path = "/media/eric/Data_3/VLA/IC1613/"
# data_path = "/Users/eric/Data/"

cube = SpectralCube.read(os.path.join(data_path, "IC1613_NA_ICL001.fits"))

# Remove empty channels
cube = cube[38:63, 500:1500, 500:1500]

# Test on some of the central channels
# test_cube = np.empty((25, 1000, 1000), dtype='uint8')
# peaks = {}
bubble_props = np.empty((0, 6), dtype=float)
region_props = {}

sigma = sig_clip(cube[0].value, nsig=10)

# cube = cube.with_mask(cube > 3*sigma*u.Jy)

for i in ProgressBar(cube.shape[0]):
    bub = BubbleFinder2D(cube[i])
    # bub.apply_bilateral_filter()
    bub.multiscale_bubblefind(sigma=sigma, overlap_frac=0.7)
    bub.region_rejection(value_thresh=3*sigma)
    # test_cube[i] = bub.bubble_mask[1]
    if bub.num_regions == 0:
        continue
    props_w_channel = \
        np.hstack([np.ones(bub.num_regions)[:, np.newaxis] * i,
                   bub.region_params])
    bubble_props = np.vstack([bubble_props, props_w_channel])
    # print(i, props_w_channel.shape[0])
    # region_props[i] = bub.region_props

# Now test clustering

# sims = pdist(bubble_props, metric=overlap_metric)
# sims[sims < 0] = 0.0
# link_mat = hier.linkage(1 - sims, 'complete')
# cluster_idx = hier.fcluster(link_mat, 0.9, criterion='distance')
# radii = np.unique(bubble_props[:, 4])

cluster_idx = hier.fclusterdata(bubble_props[:, 1:3], 18, criterion='distance',
                                method='complete')

# from sklearn.cluster import DBSCAN
# cluster_idx = DBSCAN(eps=18, min_samples=3,
#                      metric='euclidean').fit(bubble_props[:, 1:3]).labels_

ax = p.subplot(111)

# Show the moment 0
mom0 = fits.getdata(os.path.join(data_path, "IC1613_NA_X0_P_R.fits")).squeeze()
ax.imshow(mom0[500:1500, 500:1500],
          cmap='afmhot')

# ax.imshow(cube[10].value, cmap='afmhot')

cols = ['b', 'g', 'c', 'm', 'r', 'y']

i = 0
for idx in np.unique(cluster_idx[cluster_idx >= 0]):
    # if idx in remove_idx:
    #     continue
    total = bubble_props[cluster_idx == idx].shape[0]
    if total > 2:
        # print idx, total
        for blob in bubble_props[cluster_idx == idx]:
            chan, y, x, rmaj, rmin, pa = blob
            c = Ellipse((x, y), width=2*rmaj, height=2*rmin,
                        angle=np.rad2deg(pa),
                        color=cols[i % len(cols)], fill=False, linewidth=2)
            ax.add_patch(c)
            ax.plot(x, y, cols[i % len(cols)]+'D')
        x = np.mean(bubble_props[cluster_idx == idx, 2])
        y = np.mean(bubble_props[cluster_idx == idx, 1])
        ax.text(x, y, str(idx), color=cols[i % len(cols)])
        i += 1

p.xlim([0, cube.shape[2]])
p.ylim([0, cube.shape[1]])

# num_channels = (test_cube > 0).sum(0)
# clean_cube = (test_cube > 0).astype(int)

# forward = np.roll(clean_cube, 1, axis=0) - clean_cube
# backward = np.roll(clean_cube, -1, axis=0) - clean_cube

# extents = np.logical_or(forward == -1, backward == -1)

# has_one = np.any(extents, axis=0)

# for y, x in izip(*np.where(has_one)):
#     chan_diff = np.diff(np.where(extents[:, y, x]))
#     if not np.any(chan_diff):
#         # Remove that object
#         chan = np.where(extents[:, y, x])[0][0]
#         lab = test_cube[chan, y, x]
#         clean_cube[chan, np.where(test_cube[chan] == lab)] = 0
