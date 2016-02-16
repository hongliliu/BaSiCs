
import numpy as np
import matplotlib.pyplot as p
from spectral_cube import SpectralCube
import os
import skimage.morphology as mo
import skimage.measure as me
from skimage.feature import peak_local_max
from skimage.filters import rank
import scipy.ndimage as nd
from astropy.convolution import MexicanHat2DKernel, convolve_fft
from astropy.utils.console import ProgressBar
from astropy import units as u

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


# data_path = "/media/eric/Data_3/VLA/IC1613/"
data_path = "/Users/eric/Data/"

cube = SpectralCube.read(os.path.join(data_path, "IC1613_NA_ICL001.fits"))

# Remove empty channels
cube = cube[38:65, 500:1500, 500:1500]

bubble_15 = BubbleSegment(cube[15])
bubble_15.apply_bilateral_filter()
bubble_15.multiscale_bubblefind()
# bubble_15.apply_atan_transform(np.percentile(bubble_15.array, 90))

cols = ['b', 'g', 'c', 'm', 'r', 'k']

p.ion()

p.imshow(bubble_15.array, cmap='afmhot')
p.xlabel([])
p.ylabel([])
for key, labels, col in zip(np.sort(bubble_15.peaks_dict.keys()),
                            bubble_15.bubble_mask, cols):
    print key
    # p.plot(bubble_15.peaks_dict[key][:, 1],
    #        bubble_15.peaks_dict[key][:, 0], col+'D')
    cents = np.array([region.weighted_centroid for region in bubble_15.region_props[key]])
    p.plot(cents[:, 1], cents[:, 0], col+"^")

    for region in bubble_15.region_props[key]:
        ellip = me.EllipseModel()
        posns = ellip.predict_xy(np.linspace(0, 2*np.pi, 360),
                                 params=(region.centroid[0],
                                         region.centroid[1],
                                         region.major_axis_length/2.,
                                         region.minor_axis_length/2.,
                                         region.orientation+np.pi/2.))
        p.plot(posns[:, 1], posns[:, 0], col+"--")

    for i in range(1, labels.max()+1):
        p.contour(labels == i, colors=col)
p.xlim([0, bubble_15.array.shape[1]])
p.ylim([0, bubble_15.array.shape[0]])
