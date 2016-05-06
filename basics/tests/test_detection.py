
# from ._testing_data import test_gray_holes
from _testing_data import add_holes, add_gaussian_holes, shell_model

from basics.bubble_segment2D import BubbleFinder2D
from radio_beam import Beam
import numpy as np
from astropy import wcs
from astropy.convolution import convolve_fft
from spectral_cube.lower_dimensional_structures import Projection


def test_random_gray_holes():
    np.random.seed(375467546)
    gray_holes = add_holes((500, 500), hole_level=100, nholes=20,
                           max_corr=0.1, rad_max=40)
    test_gray_holes = Projection(gray_holes, wcs=wcs.WCS())

    test_bubble = BubbleFinder2D(test_gray_holes, beam=Beam(10), sigma=10,
                                 channel=0)
    test_bubble.multiscale_bubblefind(edge_find=True, nsig=5, use_ransac=True)

    test_bubble.visualize_regions(edges=True)

    print(test_bubble.region_params)


def test_single_gray_hole():
    one_gray_hole = add_holes((200, 200), hole_level=100,
                              nholes=1)
    test_gray_hole = Projection(one_gray_hole, wcs=wcs.WCS())

    test_bubble = BubbleFinder2D(test_gray_hole, beam=Beam(10), channel=0,
                                 sigma=40)
    test_bubble.multiscale_bubblefind(edge_find=False)


def test_gauss_hole():
    one_gauss_hole, params = add_gaussian_holes(np.ones((200, 200)), nholes=1,
                                                return_info=True)
    test_gauss_hole = Projection(one_gauss_hole, wcs=wcs.WCS())

    test_bubble = BubbleFinder2D(test_gauss_hole, beam=Beam(10), channel=0,
                                 sigma=0.05)
    test_bubble.multiscale_bubblefind(edge_find=True)

    test_bubble.visualize_regions(edges=True)


def test_shell():
    yy, xx = np.mgrid[-200:200, -200:200]

    beam = Beam(10)

    clean_model = shell_model(amp=1.0, ring_small=None, ratio=1.05)(yy, xx)

    amp = clean_model.max()
    sigma = 0.25 * amp

    noisy_model = clean_model + sigma * np.random.random(yy.shape)
    # smooth_model = convolve_fft(noisy_model, beam.as_kernel(1))

    model = Projection(noisy_model, wcs=wcs.WCS())

    test_bubble = BubbleFinder2D(model, beam=beam, channel=0, sigma=sigma)
    test_bubble.multiscale_bubblefind(edge_find=True)

    ax = test_bubble.visualize_regions(edges=True, show=False)
    ax.contour(model > 3 * sigma, colors='c')

    print(test_bubble.region_params)


def test_multiple_shell():
    yy, xx = np.mgrid[-200:200, -200:200]

    beam = Beam(10)

    clean_model = shell_model(amp=1.0, ring_small=None, ratio=1.05)(yy, xx)

    amp = clean_model.max()
    sigma = 0.25 * amp

    noisy_model = clean_model + sigma * np.random.random(yy.shape)
    # smooth_model = convolve_fft(noisy_model, beam.as_kernel(1))

    model = Projection(noisy_model, wcs=wcs.WCS())

    test_bubble = BubbleFinder2D(model, beam=beam, channel=0, sigma=sigma)
    test_bubble.multiscale_bubblefind(edge_find=True)

    ax = test_bubble.visualize_regions(edges=True, show=False)
    ax.contour(model > 3 * sigma, colors='c')

    print(test_bubble.region_params)

if __name__ == "__main__":
    test_random_gray_holes()
