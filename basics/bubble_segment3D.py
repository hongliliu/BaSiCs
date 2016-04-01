
import numpy as np
import astropy.units as u
from spectral_cube import SpectralCube
from astropy.utils.console import ProgressBar
import sys
from itertools import chain, izip, repeat

from bubble_segment2D import BubbleFinder2D
from bubble_objects import Bubble3D
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

    def get_bubbles(self, verbose=True, overlap_frac=0.9, min_channels=3,
                    multiprocess=True, **kwargs):
        '''
        Perform segmentation on each channel, then cluster the results to find
        bubbles.
        '''

        if verbose:
            output = sys.stdout
        else:
            output = None

        twod_regions = ProgressBar.map(_region_return,
                                       [(self.cube[i],
                                         self.cube.mask.include(view=(i, )),
                                         i, self.sigma, overlap_frac) for i in
                                        xrange(self.cube.shape[0])],
                                       multiprocess=multiprocess,
                                       file=output,
                                       step=self.cube.shape[0])

        # Join into one long list
        twod_regions = list(chain(*twod_regions))

        if len(twod_regions) == 0:
            Warning("No bubbles found in the given cube.")
            self._bubbles = []
            return

        bubble_props = np.vstack([bub.params for bub in twod_regions])

        cluster_idx = cluster_and_clean(bubble_props, **kwargs)

        self._bubbles = []

        for idx in np.unique(cluster_idx[cluster_idx >= 0]):
            clust = twod_regions[cluster_idx == idx]
            if len(clust) < min_channels:
                continue
            chans = np.array([reg.channel_center for reg in clust])

            if chans.max() + 1 - chans.min() >= min_channels:
                regions = [twod_regions[idx] for idx in
                           np.where(cluster_idx == idx)[0]]
                self._bubbles.append(Bubble3D.from_2D_regions(regions))

        return bubble_props, cluster_idx

    @property
    def bubbles(self):
        return self._bubbles


def _region_return(imps):
    arr, mask, i, sigma, overlap_frac = imps
    return BubbleFinder2D(arr, channel=i,
                          mask=mask).\
        multiscale_bubblefind(sigma=sigma,
                              overlap_frac=overlap_frac).regions
