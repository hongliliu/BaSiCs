
from spectral_cube import SpectralCube
import numpy as np
from astropy.io import fits
import astropy.units as u
import os
import matplotlib.pyplot as p
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as hier

from basics.bubble_segment3D import BubbleFinder
from basics.utils import sig_clip
from basics.log import overlap_metric

data_path = "/media/eric/Data_3/VLA/IC1613/"
# data_path = "/media/eric/MyRAID/M33/14B-088/HI/combined_HI"
# data_path = "/Users/eric/Data/"

cube = SpectralCube.read(os.path.join(data_path, "IC1613_NA_ICL001.fits"))
# cube = SpectralCube.read(os.path.join(data_path, "M33_14B-088_AT0206_HI.clean.image.fits"))

# Remove empty channels
cube = cube[38:63, 500:1500, 500:1500]
# cube = cube[48:55, 500:1500, 500:1500]
# cube = cube[:, 960:3400, 1200:2600]

bub_find = BubbleFinder(cube)

bubble_props, cluster_idx = \
    bub_find.get_bubbles(verbose=True, overlap_frac=0.5)


# from sklearn.cluster import DBSCAN
# cluster_idx = DBSCAN(eps=18, min_samples=3,
#                      metric='euclidean').fit(bubble_props[:, 1:3]).labels_

ax = p.subplot(111)

# Show the moment 0
# mom0 = cube.with_mask(cube>3*bub_find.sigma*u.Jy).moment0()
# ax.imshow(mom0.value, cmap='afmhot')
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
            y, x, rmaj, rmin, pa, chan = blob
            c = Ellipse((x, y), width=2*rmaj, height=2*rmin,
                        angle=np.rad2deg(pa),
                        color=cols[i % len(cols)], fill=False, linewidth=2)
            ax.add_patch(c)
            ax.plot(x, y, cols[i % len(cols)]+'D')
        x = np.mean(bubble_props[cluster_idx == idx, 1])
        y = np.mean(bubble_props[cluster_idx == idx, 0])
        ax.text(x, y, str(idx), color=cols[i % len(cols)])
        i += 1

p.xlim([0, cube.shape[2]])
p.ylim([0, cube.shape[1]])

p.figure()

ax = p.subplot(111)
ax.imshow(mom0[500:1500, 500:1500],
          cmap='afmhot')
# ax.imshow(mom0.value,
#           cmap='afmhot')

i = 0
for idx in np.unique(cluster_idx[cluster_idx >= 0]):
    # if idx in remove_idx:
    #     continue
    total = bubble_props[cluster_idx == idx].shape[0]
    if total <= 2:
        # print idx, total
        for blob in bubble_props[cluster_idx == idx]:
            y, x, rmaj, rmin, pa, chan = blob
            c = Ellipse((x, y), width=2*rmaj, height=2*rmin,
                        angle=np.rad2deg(pa),
                        color=cols[i % len(cols)], fill=False, linewidth=2)
            ax.add_patch(c)
            ax.plot(x, y, cols[i % len(cols)]+'D')
        x = np.mean(bubble_props[cluster_idx == idx, 1])
        y = np.mean(bubble_props[cluster_idx == idx, 0])
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
