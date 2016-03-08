
import numpy as np
import astropy.units as u
from spectral_cube import SpectralCube
from astropy.utils.console import ProgressBar


from bubble_segment2D import BubbleFinder2D
from clustering import cluster_and_clean
from utils import sig_clip


class BubbleFinder(object):
    """docstring for BubbleFinder"""
    def __init__(self, cube, wcs=None, mask=None, sigma=None, empty_channel=0):
        super(BubbleFinder, self).__init__()

        if not isinstance(cube, SpectralCube):
            if wcs is None:
                raise TypeError("When cube is not a SpectralCube, wcs must be"
                                " given.")
            cube = SpectralCube(cube, wcs)

        self.cube = cube

        self.empty_channel = empty_channel

        if sigma is None:
            self.estimate_sigma()
        else:
            self.sigma = sigma

    @property
    def cube(self):
        return self._cube

    @cube.setter
    def cube(self, input_cube):
        if input_cube.ndim != 3:
            raise TypeError("Given cube must be 3D.")

        self._cube = input_cube

    def estimate_sigma(self, nsig=10):
        '''
        Use empty channels to estimate sigma. Uses iterative sigma clipping
        to obtain a robust estimate.
        '''

        self.sigma = sig_clip(self.cube[self.empty_channel], nsig=nsig)

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, val):
        if val < 0.0:
            raise ValueError("sigma must be positive.")

        self._sigma = val

    def get_bubbles(self, verbose=True, overlap_frac=0.9, **kwargs):
        '''
        Perform segmentation on each channel, then cluster the results to find
        bubbles.
        '''

        bubble_props = np.empty((0, 6), dtype=float)

        if verbose:
            iterate = ProgressBar(self.cube.shape[0])
        else:
            iterate = xrange(self.cube.shape[0])

        for i in iterate:
            bub = BubbleFinder2D(self.cube[i],
                                 mask=self.cube.mask.include(view=(i, )))
            bub.multiscale_bubblefind(sigma=self.sigma,
                                      overlap_frac=overlap_frac)
            bub.region_rejection(value_thresh=3*self.sigma)
            if bub.num_regions == 0:
                continue
            props_w_channel = \
                np.hstack([np.ones(bub.num_regions)[:, np.newaxis] * i,
                           bub.region_params])
            bubble_props = np.vstack([bubble_props, props_w_channel])

        cluster_idx = cluster_and_clean(bubble_props, **kwargs)

        return bubble_props, cluster_idx
