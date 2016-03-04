
import numpy as np
import matplotlib.pyplot as p
from matplotlib.patches import Ellipse
from spectral_cube import SpectralCube
import os
import scipy.ndimage as nd

# import cv2

execfile("basics/bubble_segment.py")
execfile("basics/iterative_watershed.py")


def combine_scales(array, scales):
    output = np.zeros((array.shape[1], array.shape[2]))
    for i in range(array.shape[0]):
        scaled_arr = array[i]# /np.nanmax(array[0])
        output = np.nansum(np.dstack([output, scaled_arr]), axis=2)
    return output


def plot_props(img, resp, coords, scale, resp_frac=0.5):

    y, x = coords

    p.imshow(img, cmap='afmhot', origin='lower')

    # Local max
    p.plot(x, y, 'bD')

    p.contour(resp > resp[y, x]*resp_frac, colors='g')

    # Laplacian crossing points
    # p.contour(np.abs(nd.morphological_laplace(resp, footprint=mo.disk(scale)))<1, colors='r')
    laplace = nd.gaussian_laplace(img, scale)
    thresh = np.ptp(laplace) * 0.03

    p.contour(np.abs(laplace) < thresh, colors='r')

    # High gradient
    # grad = nd.gaussian_gradient_magnitude(img, scale)
    # grad_thresh = np.percentile(grad, 95)

    # p.contour(grad >= grad_thresh, colors='m')

    # Max distance
    peak_arr = np.ones_like(img, dtype='bool')
    peak_arr[y, x] = 0
    p.contour(nd.distance_transform_edt(peak_arr) <= 1.177*scale, colors='b')


data_path = "/media/eric/Data_3/VLA/IC1613/"
# data_path = "/Users/eric/Data/"

cube = SpectralCube.read(os.path.join(data_path, "IC1613_NA_ICL001.fits"))

# Remove empty channels
cube = cube[38:65, 500:1500, 500:1500]

# Find sigma in an empty channel
sigma = sig_clip(cube[0].value, nsig=10)

bubble_15 = BubbleSegment(cube[15])
# bubble_15.apply_bilateral_filter()
bubble_15.multiscale_bubblefind(sigma=sigma, overlap_frac=0.98)
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


# p.clf()
# # Exploring the gradient

# # atan = np.arctan(bubble_15.array/np.percentile(bubble_15.array, 85))
# atan = bubble_15.array

# for j, (key, labels, col) in enumerate(zip(np.sort(bubble_15.peaks_dict.keys()),
#                             bubble_15.bubble_mask, cols)):

#     # grad = nd.gaussian_gradient_magnitude(atan, 4*2**j)

#     # p.imshow(grad, cmap='afmhot')
#     p.imshow(bubble_15.array, cmap='afmhot')
#     p.colorbar()
#     # adapt = threshold_adaptive(grad, 2*4*2**j)
#     # glob = grad > np.percentile(grad, 50)
#     # mask = adapt*glob
#     # p.contour(mask, colors='r')

#     print key
#     p.plot(bubble_15.peaks_dict[key][:, 1],
#            bubble_15.peaks_dict[key][:, 0], col+'D')

#     for i in range(1, labels.max()+1):
#         peak = bubble_15.peaks_dict[key][i-1]
#         peak_val = bubble_15.wave[j, peak[0], peak[1]]
#         p.contour(labels == i, colors=col)
#         p.contour(np.logical_and(labels == i,
#                                  bubble_15.wave[j] >= peak_val/2.),
#                   colors='k')
#         p.contour(np.logical_and(labels == i,
#                                  bubble_15.wave[j] >= 0.75*peak_val),
#                   colors='k')
#         p.contour(np.logical_and(labels == i,
#                                  bubble_15.wave[j] >= 0.9*peak_val),
#                   colors='k')

#     raw_input(str(j))
#     p.clf()
