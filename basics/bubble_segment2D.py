
import numpy as np
from astropy.nddata.utils import extract_array, overlap_slices
import astropy.units as u
from warnings import warn, catch_warnings, filterwarnings
import scipy.ndimage as nd
import skimage.morphology as mo
from skimage.segmentation import clear_border
from skimage.filters import threshold_adaptive
from copy import copy
from skimage.morphology import convex_hull_image

from spectral_cube.lower_dimensional_structures import LowerDimensionalObject

from utils import sig_clip
from bubble_objects import Bubble2D
from log import blob_log, _prune_blobs, overlap_metric
from bubble_edge import find_bubble_edges
from fit_models import fit_region
from masking_utils import (smooth_edges, remove_spurs,
                           fill_nans_with_noise, fraction_in_mask)


class BubbleFinder2D(object):
    """
    Image segmentation for bubbles in a 2D image.
    """
    def __init__(self, array, scales=None, sigma=None, channel=None,
                 mask=None, cut_to_box=False, structure="beam",
                 beam=None, wcs=None, unit=None, auto_cut=False,
                 ignore_warnings=True):

        if isinstance(array, LowerDimensionalObject):
            self.array = array.value
            self.unit = array.unit
            self.wcs = array.wcs

            if 'beam' in array.meta:
                self.beam = array.meta['beam']
            elif beam is not None:
                self.beam = beam
            else:
                raise KeyError("No 'beam' in metadata. Must manually specify "
                               "the beam with the beam keyword.")

        elif isinstance(array, np.ndarray):
            self.array = array

            if beam is not None:
                self.beam = beam
            else:
                raise KeyError("Must specify the beam with the beam keyword.")

            if wcs is not None:
                self.wcs = wcs
            else:
                raise KeyError("Must specify the wcs with the wcs keyword.")

            self.unit = unit

        if sigma is None:
            # Sigma clip the array to estimate the noise level
            self.sigma = sig_clip(self.array, nsig=10)
        else:
            self.sigma = sigma

        # Set to avoid computing an array with nothing in it
        self._empty_mask_flag = False

        self.ignore_warnings = ignore_warnings

        pixscale = np.abs(self.wcs.pixel_scale_matrix[0, 0])
        fwhm_beam_pix = self.beam.major.value / pixscale
        self.beam_pix = np.ceil(fwhm_beam_pix / np.sqrt(8 * np.log(2)))

        if mask is None:
            # Should pass kwargs here
            self.create_mask()
        else:
            self.mask = mask

        if scales is None:
            # Scales incremented by sqrt(2)
            # The edge finding allows regions to expand by this factor to
            # ensure all scales are covered in between
            self.scales = self.beam_pix * \
                np.arange(1., 8 + np.sqrt(2), np.sqrt(2))
        else:
            self.scales = scales

        self._orig_shape = copy(self.array.shape)
        self.channel = channel

        if auto_cut:
            # Note: I'm still a little unsure on how much padding is necessary
            # Since the scales map to gaussians by a sqrt(2) factor, 3 times
            # should be about double the maximum kernel width, and that seems
            # like it should be enough.
            self.pad_size = 3 * np.floor(self.scales[-1]).astype(int)
            # Also pass kwargs here (some cross-over with create_mask)
            self.cut_to_bounding_box()
        else:
            self.array = np.nan_to_num(self.array)
            self._center_coords = (self._orig_shape[0] / 2,
                                   self._orig_shape[1] / 2)
            self.pad_size = 0

        # Default relative weightings for finding local maxima.
        self.weightings = np.ones_like(self.scales)
        # If searching at the beam size, decrease it's importance to
        # remove spurious features.
        # if self.scales[0] == self.beam_pix:
        #     # 0.8 removes small spurious features in the IC1613 cube
        #     # Will need to run a proper noise test to better determine
        #     # what it should be set to
        #     self.weightings[0] = 0.8

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask_array):
        if mask_array is None:
            self._mask = np.ones_like(mask_array).astype(bool)
        else:
            if mask_array.shape != self.array.shape:
                raise TypeError("mask must match the shape of the given "
                                "array.")
            self._mask = mask_array

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, input_array):
        if input_array.ndim != 2:
            raise TypeError("Given array must be 2D.")

        self._array = input_array

    @property
    def pad_size(self):
        return self._pad_size

    @pad_size.setter
    def pad_size(self, value):
        if value < 0:
            raise ValueError("Pad size must be >=0")
        self._pad_size = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):

        if value is None:
            value = self.array.min()

        if hasattr(value, 'unit'):
            value = value.value

        if value < 0.0 or ~np.isfinite(value):
            raise TypeError("sigma cannot be negative.")

        # if not isinstance(value, u.Quantity):
        #     raise TypeError("Threshold must be an astropy Quantity.")

        # if value.unit not in self.unit.find_equivalent_units():
        #     raise u.UnitsError("Threshold must have equivalent units"
        #                        " as the array " + str(self.unit))

        self._sigma = value

    def create_mask(self, bkg_nsig=3, region_min_nsig=6, adap_patch=None,
                    median_radius=None, edge_smooth_radius=None,
                    min_pixels=None, fill_radius=None,
                    mask_clear_border=True):
        '''
        Create the adaptive thresholded mask, which defines potential bubble
        edges.
        '''

        if bkg_nsig >= region_min_nsig:
            raise ValueError("bkg_nsig must be less than region_min_nsig.")

        # If parameters aren't given, set them automatically based on the
        # beam size.
        if median_radius is None:
            # Round up from 0.75 of the beam, min of 3
            median_radius = max(3, int(np.ceil(0.75 * self.beam_pix)))

        if edge_smooth_radius is None:
            # 2 pixels larger than median_radius
            edge_smooth_radius = median_radius + 2

        if fill_radius is None:
            # Same as the median radius
            fill_radius = median_radius

        if adap_patch is None:
            # ~10x the beam width (not radius) seems to be a good choice
            # My testing shows that the final mask isn't very sensitive
            # to patch changes, so long as they aren't too small or too
            # large. Mostly this is due to the rather sharp edges in the
            # shells.
            adap_patch = 10 * 2 * self.beam_pix  # 2 since this is beam radius
            # adap_patch = 10 * self.beam_pix

            # Patches must be odd.
            if adap_patch % 2 == 0:
                adap_patch -= 1

        if min_pixels is None:
            # ~ Beam size. The reconstruction will make any true beam-sized
            # objects simply the beam, so it seems safe to require ~1.5x
            # for real features
            min_pixels = int(np.floor(1.5 * np.pi * self.beam_pix ** 2))

        # Raise some warnings if the user provides large smoothing elements

        if median_radius > self.beam_pix:
            warn("It is not recommended to use a median filter larger"
                 " than the beam!")
        if fill_radius > self.beam_pix:
            warn("It is not recommended to use a median filter larger"
                 " than the beam!")

        with catch_warnings():
            if self.ignore_warnings:
                filterwarnings("ignore",
                               r"Only one label")

            medianed = nd.median_filter(self.array,
                                        footprint=mo.disk(median_radius))
            glob_mask = self.array > bkg_nsig * self.sigma
            if not glob_mask.any():
                warn("No values in the array are above the background "
                     "threshold. The mask is empty.")
                self._empty_mask_flag = True
                self.mask = glob_mask
                return

            orig_adap = threshold_adaptive(medianed, adap_patch)
            # Smooth the edges on small scales and remove small regions
            adap_mask = ~smooth_edges(glob_mask * orig_adap,
                                      edge_smooth_radius,
                                      min_pixels)
            # We've flipped the mask, so now remove small "objects"
            adap_mask = mo.remove_small_holes(adap_mask, connectivity=2,
                                              min_size=min_pixels)
            # Finally, fill in regions whose distance from an edge is
            # small (ie. unimportant 1-2 pixel cracks)
            adap_mask = remove_spurs(adap_mask, min_distance=fill_radius)

        # Remove any region which does not have a peak >5 sigma
        labels, num = nd.label(~adap_mask, np.ones((3, 3)))

        # Finally remove all mask holes (i.e. regions with signal) if if
        # does not contain a significantly bright peak (default to ~5 sigma)
        maxes = nd.maximum(self.array, labels, range(1, num + 1))
        for idx in np.where(maxes < region_min_nsig * self.sigma)[0]:
            adap_mask[labels == idx + 1] = True

        if mask_clear_border:
            adap_mask = ~clear_border(~adap_mask)

        if adap_mask.all():
            if not self.ignore_warnings:
                warn("No significant regions were found by the adaptive "
                     "thresholding. Try lowering the minimum peak required for"
                     " regions (region_min_nsig)")
            self._empty_mask_flag = True

        self.mask = adap_mask

    def cut_to_bounding_box(self, bkg_nsig=3):
        '''
        Reduce the array down to the minimum size based on the mask.
        '''

        if self._empty_mask_flag:
            if not self.ignore_warnings:
                warn("The mask is empty.")
            # Set the center_coords to the default then
            self._center_coords = (self._orig_shape[0] / 2,
                                   self._orig_shape[1] / 2)
            return

        # Not mask, since the mask is for holes.
        yslice, xslice = nd.find_objects(~self.mask)[0]

        yextent = int(yslice.stop - yslice.start)
        xextent = int(xslice.stop - xslice.start)

        self._center_coords = (int(yslice.start) + (yextent / 2),
                               int(xslice.start) + (xextent / 2))

        cut_shape = (yextent + 2 * self.pad_size,
                     xextent + 2 * self.pad_size)

        cut_arr = extract_array(self.array, cut_shape, self.center_coords,
                                mode='partial', fill_value=np.NaN)
        # Fill the NaNs with samples from the noise distribution
        # This does take the correlation of the beam out... this is fine for
        # the time being, but adding a quick convolution w/ the beam will make
        # this "proper".
        all_noise = self.array <= bkg_nsig * self.sigma
        nans = np.isnan(cut_arr)
        samps = np.random.random_integers(0, all_noise.sum() - 1,
                                          size=nans.sum())
        cut_arr[nans] = self.array[all_noise][samps]

        cut_mask = extract_array(self.mask, cut_shape, self.center_coords,
                                 mode='partial', fill_value=True)

        self.array = cut_arr
        self.mask = cut_mask

    @property
    def center_coords(self):
        return self._center_coords

    def insert_in_shape(self, array, shape, fill_value=True, dtype=np.float):
        '''
        Insert the cut down mask into the given shape.
        '''

        full_size = np.ones(shape, dtype=dtype) * fill_value

        large_slices, small_slices = \
            overlap_slices(shape, array.shape,
                           self._center_coords, mode='partial')

        full_size[large_slices] = array[small_slices]

        return full_size

    def multiscale_bubblefind(self, scales=None, nsig=2, region_pruning=False,
                              overlap_frac=0.6, edge_find=True,
                              edge_loc_bkg_nsig=3, max_eccent=3,
                              ellfit_thresh={"min_shell_frac": 0.5,
                                             "min_angular_std": 0.7},
                              max_rad=2.0, min_shell_frac=0.3, verbose=False,
                              use_ransac=False, ransac_trials=50,
                              fit_iterations=3, min_in_mask=0.75,
                              distance=None):
        '''
        Run find_bubbles on the specified scales.
        '''

        if scales is not None:
            self.scales = scales

        # Make the convex hull once.
        conv_hull = \
            nd.binary_fill_holes(
                nd.binary_dilation(~self.mask, mo.disk(10 * self.beam_pix)))
        # conv_hull = convex_hull_image(~self.mask)

        all_props = []
        all_coords = []
        for i, props in enumerate(blob_log(self.array,
                                  sigma_list=self.scales,
                                  overlap=None,
                                  threshold=nsig * self.sigma,
                                  weighting=self.weightings)):
            response_value = props[-1]

            # Adjust the region properties based on where the bubble edges are
            if edge_find:
                # Edges are defined here by the mask. value_thresh is used
                # in the case where the mask is given
                coords, shell_frac, angular_std, value_thresh = \
                    find_bubble_edges(self.array, props, max_extent=1.35,
                                      value_thresh=(nsig + 1) * self.sigma,
                                      nsig_thresh=edge_loc_bkg_nsig,
                                      return_mask=False,
                                      edge_mask=self.mask)
                # find_bubble_edges calculates the shell fraction
                # If it is below the given fraction, we skip the region.
                if len(coords) < 4:
                    if verbose:
                        print("Skipping %s" % (str(i)))
                        print(coords)
                    continue

                fail_fit = False

                for niter in range(fit_iterations):
                    coords = np.array(coords)
                    try_fit_ellipse = \
                        shell_frac >= ellfit_thresh["min_shell_frac"] and \
                        angular_std >= ellfit_thresh["min_angular_std"]

                    if niter == 0 and fit_iterations > 1:
                        iter_min_in_mask = 0.2
                    else:
                        iter_min_in_mask = min_in_mask

                    props, resid = \
                        fit_region(coords, initial_props=props,
                                   try_fit_ellipse=try_fit_ellipse,
                                   use_ransac=use_ransac,
                                   ransac_trials=ransac_trials,
                                   beam_pix=self.beam_pix, max_rad=max_rad,
                                   max_eccent=max_eccent,
                                   min_in_mask=iter_min_in_mask,
                                   mask=self.mask,
                                   image_shape=self.array.shape,
                                   max_resid=2 * self.beam_pix,
                                   conv_hull=conv_hull,
                                   verbose=verbose)

                    # Check if the fitting failed. If it did, continue on
                    if props is None:
                        fail_fit = True
                        break

                    # Now re-run the shell finding to update the coordinates
                    # with the new model.
                    coords, shell_frac, angular_std = \
                        find_bubble_edges(self.array, props, max_extent=1.05,
                                          value_thresh=value_thresh,
                                          nsig_thresh=edge_loc_bkg_nsig,
                                          try_local_bkg=False,
                                          edge_mask=self.mask)[:-1]

                    if len(coords) < 4:
                        fail_fit = True
                        break

                    if not niter == 0:
                        corr = overlap_metric(old_props, props,
                                              return_corr=True)
                        if corr >= 0.95:
                            break

                        # If the shell fraction went down, stop and revert to
                        # the last iteration
                        if old_shell_frac >= shell_frac:
                            props = old_props
                            shell_frac = old_shell_frac
                            coords = old_coords
                            angular_std = old_angular_std
                            resid = old_resid
                            break

                    old_props = props.copy()
                    old_shell_frac = shell_frac
                    old_coords = coords
                    old_angular_std = angular_std
                    old_resid = resid

                # Must satisfy the shell fraction and the full min_in_mask,
                # which may not be tested if it breaks after the first
                # iterations
                if not fail_fit:
                    final_constraints = shell_frac < min_shell_frac or \
                        fraction_in_mask(props, self.mask.T) < min_in_mask
                    # Skip and ignore
                    if final_constraints:
                        continue

                # Skip and ignore
                if fail_fit:
                    continue

            else:
                value_thresh = (nsig + 1) * self.sigma

                coords, shell_frac, angular_std = \
                    find_bubble_edges(self.array, props, max_extent=1.35,
                                      value_thresh=value_thresh,
                                      nsig_thresh=edge_loc_bkg_nsig,
                                      edge_mask=self.mask)[:-1]
                # No model, so no residual
                resid = np.NaN

            if len(coords) < 4:
                if verbose:
                    print("Skipping %s" % (str(i)))
                    print(coords)
                continue

            # Transform coordinates to the original array shape. Needed when
            # auto_cut is used (i.e., cut_to_bounding_box)
            # Defined using the center of the arrays, the transform is:
            # X = (x - x_c) + X_c
            cut_shape = self.array.shape
            props[0] = (props[0] - cut_shape[0] / 2) + self.center_coords[0]
            props[1] = (props[1] - cut_shape[1] / 2) + self.center_coords[1]

            coords = np.array(coords)
            coords[:, 0] = (coords[:, 0] - cut_shape[0] / 2) + \
                self.center_coords[0]
            coords[:, 1] = (coords[:, 1] - cut_shape[1] / 2) + \
                self.center_coords[1]

            # Append useful info onto the properties
            props = np.append(props, response_value)
            props = np.append(props, shell_frac)
            props = np.append(props, angular_std)
            props = np.append(props, resid)

            all_props.append(props)
            all_coords.append(coords)

        all_props = np.array(all_props)

        if not len(all_props) == 0:

            # First remove nearly duplicated regions. This stops much
            # smaller regions (corr <= 0.75) from dominating larger ones,
            # when they may only be a portion of the correct shape
            all_props, all_coords = \
                _prune_blobs(all_props, all_coords,
                             method="shell fraction",
                             min_corr=0.8, blob_merge=False)

            # Only keep pruning if region_pruning is enabled. If you're
            # stacking between channels, it is recommended that this be
            # disabled.
            if region_pruning:
                # Now look on smaller scales, and enable matching between
                # smaller regions embedded in a larger one. About 0.5 is
                # appropriate since a completely embedded smaller region will
                # have 1/sqrt(3) for the maximally eccentric shape allowed
                # (e~3). This does make it possible that a much larger region
                # could be lost when it shouldn't be. So long as the minimum
                # cut used in the clustering is ~0.5, these can still be
                # clustered appropriately.
                all_props, all_coords = \
                    _prune_blobs(all_props, all_coords,
                                 method="shell fraction",
                                 min_corr=overlap_frac)

                # Any highly overlapping regions should now be small regions
                # inside much larger ones. We're going to assume that the
                # remaining large regions are more important (good based on
                # by-eye evaluation). Keeping this at 0.75, since we only want
                # to remove very highly overlapping small regions. Note that
                # this does set an upper limit on how overlapped region may
                # be. This is fairly necessary though due to the shape
                # ambiguities present in assuming an elliptical shape.
                all_props, all_coords = \
                    _prune_blobs(all_props, all_coords, overlap=0.75,
                                 method='size')

            self._regions = \
                [Bubble2D(prop, shell_coords=coord, channel=self.channel,
                          distance=distance)
                 for prop, coord in zip(all_props, all_coords)]
        else:
            self._regions = []

        return self

    @property
    def regions(self):
        return self._regions

    @property
    def region_params(self):
        return np.array([bub2D.params for bub2D in self.regions])

    @property
    def num_regions(self):
        return len(self.regions)

    def visualize_regions(self, show=True, edges=False, ax=None, array=None,
                          region_col='b', edge_col='g', log_scale=False,
                          show_mask_contours=True):
        '''
        Show the regions optionally overlaid with the edges.
        '''

        if len(self.regions) == 0:
            warn("No regions were found. Nothing to show.")
            return

        import matplotlib.pyplot as p

        if ax is None:
            ax = p.subplot(111)

        if array is not None:
            full_array = array
        else:
            full_array = self.insert_in_shape(self.array, self._orig_shape,
                                              fill_value=0.0)

        if log_scale:
            ax.imshow(np.log10(full_array), cmap='afmhot', origin='lower')
        else:
            ax.imshow(full_array, cmap='afmhot', origin='lower')

        if show_mask_contours:
            ax.contour(self.insert_in_shape(self.mask, self._orig_shape),
                       colors='k')

        for bub in self.regions:
            ax.add_patch(bub.as_patch(color=region_col, fill=False,
                                      linewidth=2))
            ax.plot(bub.x, bub.y, region_col + 'D')
            if edges:
                ax.plot(bub.shell_coords[:, 1], bub.shell_coords[:, 0],
                        edge_col + "o")

        p.xlim([0, full_array.shape[1]])
        p.ylim([0, full_array.shape[0]])

        if show:
            p.show()
        else:
            return ax

    def save_regions(self, folder=None, name=None):
        '''
        Save regions as pickled objects.

        Parameters
        ----------
        folder : str, optional
            Path to where objects will be saved.
        name : str, optional
            Prefix for the save names.
        '''

        if len(self.regions) == 0:
            warn("There are no regions. Returning.")
            return

        import os

        if folder is None:
            folder = ""

        if name is None:
            name = ""

        file_prefix = os.path.join(folder, name)

        for i, reg in enumerate(self.bubbles):
            if len(file_prefix) == 0:
                save_name = "region_{}.pkl".format(i)
            else:
                save_name = "{0}_region_{1}.pkl".format(file_prefix, i)

            reg.save_bubble(save_name)
