
import numpy as np
import astropy.units as u
from spectral_cube import SpectralCube
# from astropy.utils.console import ProgressBar
from astropy.coordinates import SkyCoord, Angle
import sys
from warnings import warn

from bubble_segment2D import BubbleFinder2D
from bubble_objects import Bubble3D
from bubble_catalog import PPV_Catalog
from clustering import cluster_and_clean, cluster_brute_force
from utils import sig_clip
from progressbar import ProgressBar

GALAXY_KEYS = ["inclination", "position_angle", "center_coord",
               "scale_height"]


class BubbleFinder(object):
    """docstring for BubbleFinder"""
    def __init__(self, cube, wcs=None, mask=None, sigma=None, empty_channel=0,
                 keep_threshold_mask=False, distance=None, galaxy_props={}):
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
        self.distance = distance
        self.galaxy_props = galaxy_props

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

    @property
    def galaxy_props(self):
        return self._galaxy_props

    @galaxy_props.setter
    def galaxy_props(self, input_dict):

        # Make sure all of the kwargs are given.
        in_input = [True if key in GALAXY_KEYS else False
                    for key in input_dict]
        if not np.all(in_input):
            missing = list(set(GALAXY_KEYS) - set(in_input))
            raise KeyError("galaxy_props is missing these keys: {}"
                           .format(missing))

        if not isinstance(input_dict["center_coord"], SkyCoord):
            raise TypeError("center_coords must be a SkyCoord.")

        if not input_dict["distance"].unit.is_equivalent(u.pc):
            raise u.UnitsError("distance must have a unit of distance")

        if not input_dict["scale_height"].unit.is_equivalent(u.pc):
            raise u.UnitsError("scale_height must have a unit of distance")

        if not isinstance(input_dict["inclination"], Angle):
            raise TypeError("inclination must be an Angle.")

        if not isinstance(input_dict["position_angle"], Angle):
            raise TypeError("position_angle must be an Angle.")

        self._galaxy_props = input_dict

    def get_bubbles(self, verbose=True, overlap_frac=0.9, min_channels=3,
                    use_cube_mask=False, nsig=2., refit=False,
                    cube_linewidth=None, multiprocess=True, nprocesses=None,
                    **kwargs):
        '''
        Perform segmentation on each channel, then cluster the results to find
        bubbles.
        '''

        if verbose:
            output = sys.stdout
        else:
            output = None

        if verbose:
            print("Running bubble finding plane-by-plane.")
        twod_results = \
            ProgressBar.map(_region_return,
                            ((self.cube[i],
                              self.cube.mask.include(view=(i, ))
                              if use_cube_mask else None,
                              i, self.sigma, nsig, overlap_frac,
                              self.keep_threshold_mask, self.distance)
                             for i in xrange(self.cube.shape[0])),
                            multiprocess=multiprocess,
                            nprocesses=nprocesses,
                            file=output,
                            step=self.cube.shape[0],
                            item_len=self.cube.shape[0])

        twod_regions = []
        if self.keep_threshold_mask:
            self._mask = np.zeros(self.cube.shape, dtype=np.bool)

        for out in twod_results:
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

        # cluster_idx = cluster_and_clean(bubble_props, **kwargs)
        cluster_idx = cluster_brute_force(bubble_props, **kwargs)

        # Add the unclustered ones first
        unclusts = [twod_regions[idx] for idx in np.where(cluster_idx == 0)[0]]
        for reg in unclusts:
            self._unclustered_regions.append([reg])

        good_clusters = []
        for idx in np.unique(cluster_idx[cluster_idx > 0]):
            regions = [twod_regions[idx] for idx in
                       np.where(cluster_idx == idx)[0]]

            if len(regions) < min_channels:
                self._unclustered_regions.append(regions)
                continue

            chans = np.array([reg.channel_center for reg in regions])
            if chans.max() + 1 - chans.min() >= min_channels:
                good_clusters.append(regions)
            else:
                self._unclustered_regions.append(regions)

        if verbose:
            print("Creating bubbles and finding their properties.")
        # We need to pass a linewidth array for the cube, but don't want to
        # have to recompute it multiple times. Make sure it's using the mask
        if cube_linewidth is None:
            cube_linewidth = self.cube.with_mask(self.mask).linewidth_fwhm()
        # Now create the bubble objects and find their respective properties
        self._bubbles = ProgressBar.map(_make_bubble,
                                        ((regions, refit, self.cube, self.mask,
                                          self.distance, self.sigma,
                                          cube_linewidth, self.galaxy_props)
                                         for regions in good_clusters),
                                        multiprocess=False,
                                        nprocesses=nprocesses,
                                        file=output,
                                        step=1,
                                        item_len=len(good_clusters))

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

    def to_catalog(self):
        '''
        Returns a PPV_Catalog to explore the population properties.
        '''
        return PPV_Catalog(self.bubbles)

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
            moment0 = self.cube.moment0().value

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
                    for reg in bub.twoD_regions:
                        ax.add_patch(reg.as_patch(color=region_col,
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
                        color = region_col
                    else:
                        linewidth = 3
                        linestyle = "--"
                        color = 'm'
                    ax.add_patch(bub.as_patch(color=color, fill=False,
                                              linewidth=linewidth,
                                              linestyle=linestyle))
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
                               edges=False, plot_unclustered=False,
                               interactive_plot=True,
                               show_mask_contours=False):
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
        if interactive_plot:
            p.ion()

        for chan in chans:
            if subplot:
                raise NotImplementedError("Need subplots at some point.")
            else:
                ax = p.subplot(111)

            ax.imshow(self.cube[chan].value, cmap='afmhot', origin='lower')

            if show_mask_contours:
                if self.keep_threshold_mask:
                    ax.contour(self.mask[chan], colors='k')
                else:
                    warn("'keep_threshold_mask' must be enabled to plot mask "
                         "contours.")

            for bub in self.bubbles:
                if chan < bub.channel_start and chan > bub.channel_end:
                    continue
                for twod in bub.twoD_regions:
                    if twod.channel_center != chan:
                        continue
                    ax.add_patch(twod.as_patch(color='b', fill=False,
                                               linewidth=2))
                    ax.plot(twod.x, twod.y, 'bD')
                    if edges:
                        ax.plot(twod.shell_coords[:, 1],
                                twod.shell_coords[:, 0], "go")

            # Also show the unclustered regions in each channel
            if plot_unclustered:
                for clust in self.unclustered_regions:
                    for reg in clust:
                        if reg.channel_center != chan:
                            continue
                        ax.add_patch(reg.as_patch(color='r', fill=False,
                                                  linewidth=2))
                        ax.plot(reg.x, reg.y, 'rD')
                        if edges:
                            ax.plot(reg.shell_coords[:, 1],
                                    reg.shell_coords[:, 0], "ro")

            p.xlim([0, self.cube.shape[2]])
            p.ylim([0, self.cube.shape[1]])

            if interactive_plot:
                p.draw()
                raw_input("Channel {}".format(chan))
                p.clf()
            else:
                p.show()


def _region_return(imps):
    arr, mask, i, sigma, nsig, overlap_frac, return_mask, distance = imps
    bubs = BubbleFinder2D(arr, channel=i,
                          mask=mask, sigma=sigma).\
        multiscale_bubblefind(nsig=nsig,
                              overlap_frac=overlap_frac,
                              distance=distance)
    if return_mask:
        return i, bubs.regions, bubs.mask

    return i, bubs.regions


def _make_bubble(imps):
    regions, refit, cube, mask, distance, sigma, lwidth, galaxy_props = imps
    return Bubble3D.from_2D_regions(regions, refit=refit,
                                    cube=cube, mask=mask,
                                    distance=distance,
                                    sigma=sigma, linewidth=lwidth,
                                    galaxy_kwargs=galaxy_props)
