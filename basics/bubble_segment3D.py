
import numpy as np
import astropy.units as u
from spectral_cube import SpectralCube
from astropy.utils.console import ProgressBar
import sys
from warnings import warn

from bubble_segment2D import BubbleFinder2D
from bubble_objects import Bubble3D
from clustering import cluster_and_clean
from utils import sig_clip


class BubbleFinder(object):
    """docstring for BubbleFinder"""
    def __init__(self, cube, wcs=None, mask=None, sigma=None, empty_channel=0,
                 keep_threshold_mask=False):
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

        self.keep_threshold_mask = keep_threshold_mask
        self._mask = None

    @property
    def cube(self):
        return self._cube

    @cube.setter
    def cube(self, input_cube):
        if input_cube.ndim != 3:
            raise TypeError("Given cube must be 3D.")

        self._cube = input_cube

    @property
    def mask(self):
        if not self.keep_threshold_mask:
            warn("Enable keep_threshold_mask to access the entire mask.")

        return self._mask

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
                    multiprocess=True, use_cube_mask=False, nsig=2.,
                    refit=True, **kwargs):
        '''
        Perform segmentation on each channel, then cluster the results to find
        bubbles.
        '''

        if verbose:
            output = sys.stdout
        else:
            output = None

        output = \
            ProgressBar.map(_region_return,
                            [(self.cube[i],
                              self.cube.mask.include(view=(i, ))
                              if use_cube_mask else None,
                              i, self.sigma, nsig, overlap_frac,
                              self.keep_threshold_mask) for i in
                             xrange(self.cube.shape[0])],
                            multiprocess=multiprocess,
                            file=output,
                            step=self.cube.shape[0])

        twod_regions = []
        if self.keep_threshold_mask:
            self._mask = np.zeros(self.cube.shape, dtype=np.bool)

        for out in output:
            if self.keep_threshold_mask:
                chan, regions, mask_slice = out

                self._mask[chan] = mask_slice
            else:
                chan, regions = out

            twod_regions.extend(regions)

        self._bubbles = []
        self._unclustered_regions = []

        if len(twod_regions) == 0:
            warn("No bubbles found in the given cube.")
            return self

        bubble_props = np.vstack([bub.params for bub in twod_regions])

        cluster_idx = cluster_and_clean(bubble_props, **kwargs)

        for idx in np.unique(cluster_idx[cluster_idx >= 0]):
            regions = [twod_regions[idx] for idx in
                       np.where(cluster_idx == idx)[0]]

            if len(regions) < min_channels:
                self._unclustered_regions.append(regions)
                continue

            chans = np.array([reg.channel_center for reg in regions])
            if chans.max() + 1 - chans.min() >= min_channels:
                self._bubbles.append(Bubble3D.from_2D_regions(regions,
                                                              refit=refit,
                                                              cube=self.cube))
            else:
                self._unclustered_regions.append(regions)

        return self

    @property
    def bubbles(self):
        return self._bubbles

    @property
    def num_bubbles(self):
        return len(self.bubbles)

    @property
    def unclustered_regions(self):
        return self._unclustered_regions

    def visualize_bubbles(self, show=True, edges=False, ax=None,
                          moment0=None, region_col='b', edge_col='g',
                          log_scale=False, plot_threeD_shapes=True,
                          plot_twoD_shapes=False, plot_unclustered=False):
        '''
        Show the location of the bubbles on the moment 0 array.
        '''
        if len(self.bubbles) == 0:
            warn("No bubbles were found. Nothing to show.")
            return

        if moment0 is None:
            # Create the moment array from the cube
            moment0 = self.cube.moment0()

        import matplotlib.pyplot as p

        if ax is None:
            ax = p.subplot(111)

        if log_scale:
            ax.imshow(np.log10(moment0), cmap='afmhot', origin='lower')
        else:
            ax.imshow(moment0, cmap='afmhot', origin='lower')

        if plot_unclustered:
            for clust in self.unclustered_regions:
                for reg in clust:
                    ax.add_patch(reg.as_patch(color=region_col, fill=False,
                                              linewidth=2))
                    ax.plot(reg.x, reg.y, region_col + 'D')
                    if edges:
                        ax.plot(reg.shell_coords[:, 1], reg.shell_coords[:, 0],
                                edge_col + "o")
        else:
            for bub in self.bubbles:
                if plot_twoD_shapes:
                    for twod in bub.twoD_regions:
                        ax.add_patch(twod.as_patch(color=region_col,
                                                   fill=False,
                                                   linewidth=2))
                        ax.plot(reg.x, reg.y, region_col + "D")
                        if edges:
                            ax.plot(reg.shell_coords[:, 1],
                                    reg.shell_coords[:, 0], edge_col + "o")
                if plot_threeD_shapes:
                    if not plot_twoD_shapes:
                        linewidth = 2
                        linestyle = "-"
                    else:
                        linewidth = 3
                        linestyle = "--"
                    ax.add_patch(bub.as_patch(color=region_col, fill=False,
                                              linewidth=2, linestyle))
                    ax.plot(bub.x, bub.y, region_col + 'D')
                    if edges:
                        ax.plot(bub.shell_coords[:, 2], bub.shell_coords[:, 1],
                                edge_col + "o")

        p.xlim([0, self.cube.shape[2]])
        p.ylim([0, self.cube.shape[1]])

        if show:
            p.show()
        else:
            return ax

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
    arr, mask, i, sigma, nsig, overlap_frac, return_mask = imps
    bubs = BubbleFinder2D(arr, channel=i,
                          mask=mask, sigma=sigma).\
        multiscale_bubblefind(nsig=nsig,
                              overlap_frac=overlap_frac)
    if return_mask:
        return i, bubs.regions, bubs.mask

    return i, bubs.regions
