
import numpy as np
import astropy.units as u
from spectral_cube import SpectralCube
# from astropy.utils.console import ProgressBar
import sys
from warnings import warn
from copy import copy

from bubble_segment2D import BubbleFinder2D
from bubble_objects import Bubble3D, Bubble2D
from bubble_catalog import PPV_Catalog
from clustering import cluster_brute_force, threeD_overlaps
from utils import sig_clip
from galaxy_utils import gal_props_checker
from progressbar import ProgressBar


class BubbleFinder(object):
    """docstring for BubbleFinder"""
    def __init__(self, cube, wcs=None, mask=None, sigma=None, empty_channel=0,
                 keep_threshold_mask=True, distance=None, galaxy_props={}):
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
        gal_props_checker(input_dict)

        self._galaxy_props = input_dict

    def get_bubbles(self, verbose=True, overlap_frac=0.9, min_channels=3,
                    use_cube_mask=False, nsig=2., refit=False, scales=None,
                    cube_linewidth=None, multiprocess=True, nprocesses=None,
                    twod_regions=None, mask=None, min_shell_fraction=0.4,
                    save_regions=False, save_region_path=None,
                    overlap_kwargs={}, **kwargs):
        '''
        Perform segmentation on each channel, then cluster the results to find
        bubbles.
        '''

        if verbose:
            output = sys.stdout
        else:
            output = None

        if cube_linewidth is not None:
            if not cube_linewidth.unit.is_equivalent(u.m / u.s):
                raise u.UnitsError("cube_linewidth must have velocity units.")

        if twod_regions is None:
            if verbose:
                print("Running bubble finding plane-by-plane.")
            twod_results = \
                ProgressBar.map(_region_return,
                                ((self.cube[i],
                                  self.cube.mask.include(view=(i, ))
                                  if use_cube_mask else None,
                                  i, self.sigma, nsig, overlap_frac,
                                  self.keep_threshold_mask, self.distance,
                                  scales)
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
        else:
            for reg in twod_regions:
                if not isinstance(reg, Bubble2D):
                    raise TypeError("twod_regions must all be Bubble2D"
                                    " objects.")
            if mask is not None:
                assert mask.shape == self.cube.shape
                self._mask = mask

        if save_regions:
            import os

            if save_region_path is None:
                save_region_path = ""

            for i, reg in enumerate(twod_regions):
                reg.save_bubble(os.path.join(save_region_path,
                                             "twod_region_{}.pkl".format(i)))

        self._bubbles = []
        self._unclustered_regions = []

        if len(twod_regions) == 0:
            warn("No bubbles found in the given cube.")
            return self

        bubble_props = np.vstack([bub.params for bub in twod_regions])

        if verbose:
            print("Clustering 2D regions across channels.")
        # cluster_idx = cluster_and_clean(bubble_props, **kwargs)
        cluster_idx = cluster_brute_force(bubble_props, **kwargs)

        # Add the unclustered ones first
        unclusts = [twod_regions[idx] for idx in np.where(cluster_idx == 0)[0]]
        for reg in unclusts:
            self._unclustered_regions.append([reg])

        # Convert everything into a 3D bubble for merging/pruning
        initial_bubbles = []
        for idx in np.unique(cluster_idx[cluster_idx > 0]):
            regions = [twod_regions[idx] for idx in
                       np.where(cluster_idx == idx)[0]]

            # These bubbles won't have their physical properties set
            initial_bubbles.append(Bubble3D.from_2D_regions(regions))

        if verbose:
            print("Joining and pruning bubbles.")
        # Now we prune off overlapping bubbles
        initial_bubbles, removed_bubbles, new_twoD_clusters = \
            threeD_overlaps(initial_bubbles, **overlap_kwargs)

        # Add the 2D regions in the removed bubbles to the unclustered list
        for bub in removed_bubbles:
            self._unclustered_regions.append(bub.twoD_regions)

        # Make the joined regions into bubbles
        for regs in new_twoD_clusters:
            initial_bubbles.append(Bubble3D.from_2D_regions(regs))

        # Now we want to sort through the initial bubbles list and only
        # keep those that satisfy the channel requirement
        removals = []
        for i, bub in enumerate(initial_bubbles):
            if bub.channel_width < min_channels:
                removals.append(i)

        # Remove from the end so the indexing doesn't get messed up.
        for r in removals[::-1]:
            self._unclustered_regions.append(initial_bubbles[r].twoD_regions)
            initial_bubbles.pop(r)

        if verbose:
            print("Creating bubbles and finding their properties.")
        # We need to pass a linewidth array for the cube, but don't want to
        # have to recompute it multiple times. Make sure it's using the mask
        if cube_linewidth is None:
            # This gives some funky results
            # cube_linewidth = self.cube.with_mask(self.mask).linewidth_fwhm()
            # Just mask with a sigma cut
            sigma_w_unit = self.sigma * self.cube.unit
            cube_linewidth = \
                self.cube.with_mask(self.cube >= 3 *
                                    sigma_w_unit).linewidth_fwhm()
        # Now create the bubble objects and find their respective properties
        self._bubbles = ProgressBar.map(_make_bubble,
                                        ((bub.twoD_regions, refit, self.cube,
                                          self.mask, self.distance, self.sigma,
                                          cube_linewidth, self.galaxy_props)
                                         for bub in initial_bubbles),
                                        multiprocess=False,
                                        nprocesses=nprocesses,
                                        file=output,
                                        step=1,
                                        item_len=len(initial_bubbles))

        # Finally, we're going to prune off any bubbles whose shell fraction is
        # less than min_shell_fraction (~0.4). Recall that the 3D shell
        # fraction is the maximum shell fraction of its constituent 2D regions
        all_bubbles = copy(self.bubbles)

        self._bubbles = [bub for bub in all_bubbles if
                         bub.shell_fraction >= min_shell_fraction]

        return self

    @staticmethod
    def reload(cube, bubbles, mask=None, distance=None, galaxy_props=None):
        '''
        Reload from a cube and a list of bubbles.
        '''

        if mask is not None:
            assert cube.shape == mask.shape

        self = BubbleFinder(cube, mask=mask, distance=distance,
                            galaxy_props=galaxy_props)

        self._bubbles = bubbles
        self._unclustered_regions = []

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
        if self.num_bubbles == 0:
            warn("No bubbles were found. Cannot return a catalog.")
            return

        return PPV_Catalog(self.bubbles, self.galaxy_props)

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
                               show_mask_contours=False, start_chan=None,
                               save=False, save_path=None, save_name=None):
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

        if start_chan is not None:
            if start_chan >= self.cube.shape[0]:
                raise ValueError("start_chan must be below the number of"
                                 " channels: {}".format(self.cube.shape[0]))

            chans = chans[chans >= start_chan]

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

            if save:
                import os

                if save_path is None:
                    save_path = ""
                if save_name is None:
                    save_name = ""

                p.savefig(os.path.join(save_path,
                                       "{0}_channel_{1}.pdf".format(save_name,
                                                                    chan)))
                p.clf()
            else:
                if interactive_plot:
                    p.draw()
                    raw_input("Channel {}".format(chan))
                    p.clf()
                else:
                    p.show()

    def save_bubbles(self, folder=None, name=None):
        '''
        Save bubbles as pickled objects.

        Parameters
        ----------
        folder : str, optional
            Path to where objects will be saved.
        name : str, optional
            Prefix for the save names.
        '''

        if len(self.bubbles) == 0:
            warn("There are no bubbles. Returning.")
            return

        import os

        if folder is None:
            folder = ""

        if name is None:
            name = ""

        file_prefix = os.path.join(folder, name)

        for i, bub in enumerate(self.bubbles):
            if len(file_prefix) == 0:
                save_name = "bubble_{}.pkl".format(i)
            else:
                save_name = "{0}_bubble_{1}.pkl".format(file_prefix, i)

            bub.save_bubble(save_name)


def _region_return(imps):
    arr, mask, i, sigma, nsig, overlap_frac, return_mask, distance, scales = \
        imps
    bubs = BubbleFinder2D(arr, channel=i,
                          mask=mask, sigma=sigma, auto_cut=True,
                          scales=scales).\
        multiscale_bubblefind(nsig=nsig,
                              overlap_frac=overlap_frac,
                              distance=distance)
    if return_mask:
        return i, bubs.regions, \
            bubs.insert_in_shape(bubs.mask, bubs._orig_shape, fill_value=True,
                                 dtype=bool)

    return i, bubs.regions


def _make_bubble(imps):
    regions, refit, cube, mask, distance, sigma, lwidth, galaxy_props = imps
    return Bubble3D.from_2D_regions(regions, refit=refit,
                                    cube=cube, mask=mask,
                                    distance=distance,
                                    sigma=sigma, linewidth=lwidth,
                                    galaxy_kwargs=galaxy_props)
