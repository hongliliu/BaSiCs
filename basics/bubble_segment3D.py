
import numpy as np
import astropy.units as u
from spectral_cube import SpectralCube
from astropy.utils.console import ProgressBar
import sys
from itertools import chain

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
            if mask is not None:
                cube = cube.with_mask(mask)

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
        if val < 0.0 or ~np.isfinite(val):
            raise ValueError("sigma must be positive and finite.")

        self._sigma = val

    def get_bubbles(self, verbose=True, overlap_frac=0.9, min_channels=3,
                    multiprocess=True, use_cube_mask=False, **kwargs):
        '''
        Perform segmentation on each channel, then cluster the results to find
        bubbles.
        '''

        if verbose:
            output = sys.stdout
        else:
            output = None

        twod_regions = \
            ProgressBar.map(_region_return,
                            [(self.cube[i],
                              self.cube.mask.include(view=(i, ))
                              if use_cube_mask else None,
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
        self._unclustered_regions = []

        for idx in np.unique(cluster_idx[cluster_idx >= 0]):
            regions = [twod_regions[idx] for idx in
                       np.where(cluster_idx == idx)[0]]

            if len(regions) < min_channels:
                self._unclustered_regions.append(regions)
                continue

            chans = np.array([reg.channel_center for reg in regions])
            if chans.max() + 1 - chans.min() >= min_channels:
                self._bubbles.append(Bubble3D.from_2D_regions(regions))
            else:
                self._unclustered_regions.append(regions)

        return self

    @property
    def bubbles(self):
        return self._bubbles

    @property
    def unclustered_regions(self):
        return self._unclustered_regions

    def visualize_channel_maps(self, all_chans=False, subplot=False,
                               edges=False):
        '''
        Plot each channel optionally overlaid with the regions and/or the
        edges.
        '''

        # Loop through all, or just those that have at least one region
        if all_chans:
            chans = np.arange(self.cube.shape[0], dtype=np.int)
        else:
            chans = []
            for bub in self.bubbles:
                chans.extend(list(bub.twoD_region_params()[:, -1]))
            chans = np.unique(np.array(chans, dtype=np.int))

        import matplotlib.pyplot as p

        for chan in chans:
            if subplot:
                raise NotImplementedError("Need subplots at some point.")
            else:
                ax = p.subplot(111)

            ax.imshow(self.cube[chan].value, cmap='afmhot', origin='lower')

            for bub in self.bubbles:
                if chan < bub.channel_start and chan > bub.channel_end:
                    continue
                for twod in bub.twoD_objects:
                    if twod.channel_center != chan:
                        continue
                    ax.add_patch(twod.as_patch(color='b', fill=False,
                                               linewidth=2))
                    ax.plot(twod.x, twod.y, 'bD')
                    if edges:
                        ax.plot(twod.shell_coords[:, 1],
                                twod.shell_coords[:, 0], "go")

            p.xlim([0, self.cube.shape[2]])
            p.ylim([0, self.cube.shape[1]])

            p.show()


def _region_return(imps):
    arr, mask, i, sigma, overlap_frac = imps
    return BubbleFinder2D(arr, channel=i,
                          mask=mask, sigma=sigma).\
        multiscale_bubblefind(overlap_frac=overlap_frac).regions
