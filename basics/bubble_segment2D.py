
import numpy as np
from astropy.nddata.utils import extract_array, add_array
import astropy.units as u
from warnings import warn
import scipy.ndimage as nd
import skimage.morphology as mo
from skimage.filters import threshold_adaptive

from spectral_cube.lower_dimensional_structures import LowerDimensionalObject

from basics.utils import sig_clip
from basics.bubble_objects import Bubble2D
from basics.log import blob_log, _prune_blobs
from basics.bubble_edge import find_bubble_edges
from basics.fit_models import fit_region
from basics.masking_utils import smooth_edges, remove_spurs


class BubbleFinder2D(object):
    """
    Image segmentation for bubbles in a 2D image.
    """
    def __init__(self, array, scales=None, sigma=None, channel=None,
                 mask=None, cut_to_box=False, pad_size=0, structure="beam",
                 beam=None, wcs=None, unit=None):

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

        pixscale = np.abs(self.wcs.pixel_scale_matrix[0, 0])
        fwhm_beam_pix = self.beam.major.value / pixscale
        self.beam_pix = np.ceil(fwhm_beam_pix / np.sqrt(8 * np.log(2)))

        if mask is None:
            # Should pass kwargs here
            self.create_mask()
        else:
            self.mask = mask
        self.pad_size = pad_size

        self.array = np.nan_to_num(self.array)
        self._orig_shape = self.array.shape
        self.channel = channel

        if scales is None:
            # Scales incremented by sqrt(2)
            # The edge finding allows regions to expand by this factor to
            # ensure all scales are covered in between
            self.scales = self.beam_pix * \
                np.arange(1., 8 + np.sqrt(2), np.sqrt(2))
        else:
            self.scales = scales

        # Default relative weightings for finding local maxima.
        self.weightings = np.ones_like(self.scales)
        # If searching at the beam size, decrease it's importance to
        # remove spurious features.
        if self.scales[0] == self.beam_pix:
            # 0.8 removes small spurious features in the IC1613 cube
            # Will need to run a proper noise test to better determine
            # what it should be set to
            self.weightings[0] = 0.8

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

    def create_mask(self, bkg_nsig=2, region_min_nsig=4, adap_patch=None,
                    median_radius=None, edge_smooth_radius=None,
                    min_pixels=None, fill_radius=None):
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

        medianed = nd.median_filter(self.array,
                                    footprint=mo.disk(median_radius))
        glob_mask = self.array > bkg_nsig * self.sigma
        if not glob_mask.any():
            warn("No values in the array are above the background threshold."
                 " The mask is empty.")
            self._empty_mask_flag = True
            self.mask = glob_mask
            return

        orig_adap = threshold_adaptive(medianed, adap_patch)
        # Smooth the edges on small scales and remove small regions
        adap_mask = ~smooth_edges(glob_mask * orig_adap, edge_smooth_radius,
                                  min_pixels)
        # We've flipped the mask, so now remove small "objects"
        adap_mask = mo.remove_small_holes(adap_mask, connectivity=2,
                                          min_size=min_pixels)
        # Finally, fill in regions whose distance from an edge is small (ie.
        # unimportant 1-2 pixel cracks)
        adap_mask = remove_spurs(adap_mask, min_distance=fill_radius)

        # Remove any region which does not have a peak >5 sigma
        labels, num = nd.label(~adap_mask, np.ones((3, 3)))

        # Finally remove all mask holes (i.e. regions with signal) if if
        # does not contain a significantly bright peak (default to ~5 sigma)
        maxes = nd.maximum(self.array, labels, range(1, num + 1))
        for idx in np.where(maxes < region_min_nsig * self.sigma)[0]:
            adap_mask[labels == idx + 1] = True

        if not adap_mask.any():
            warn("No significant regions were found by the adaptive "
                 "thresholding. Try lowering the minimum peak required for "
                 "regions (region_min_nsig)")
            self._empty_mask_flag = True

        self.mask = adap_mask

    def cut_to_bounding_box(self, pad_size=0):
        '''
        Reduce the array down to the minimum size based on the mask.
        Optionally add padding to the reduced size.
        '''

        if pad_size != self.pad_size:
            self.pad_size = pad_size

        mask_pixels = np.where(self.mask)

        yedges = [mask_pixels[0].min(), mask_pixels[0].max()]
        xedges = [mask_pixels[1].min(), mask_pixels[1].max()]

        self._center_coords = ((yedges[1] + 1 + yedges[0])/2,
                               (xedges[1] + 1 + xedges[0])/2)

        cut_shape = (yedges[1]+1-yedges[0], xedges[1]+1-xedges[0])

        cut_arr = extract_array(self.array, cut_shape, self.center_coords)
        cut_mask = extract_array(self.mask, cut_shape, self.center_coords)

        if self.pad_size > 0:
            # Pads edges with zeros
            self.array = np.pad(cut_arr, self.pad_size, mode='constant')
            self.mask = np.pad(cut_mask, self.pad_size, mode='constant')
        else:
            self.array = cut_arr
            self.mask = cut_mask

    @property
    def center_coords(self):
        return self._center_coords

    def insert_in_shape(self, shape):
        '''
        Insert the cut down mask into the given shape.
        '''

        if self.bubble_mask.shape == shape:
            return self.bubble_mask
        else:
            full_size = np.zeros(shape)
            return add_array(full_size, self.bubble_mask, self.center_coords)

    @property
    def bubble_mask(self):
        return self._bubble_mask

    def multiscale_bubblefind(self, scales=None, nsig=2,
                              overlap_frac=0.6, edge_find=True,
                              edge_loc_bkg_nsig=3, max_eccent=3,
                              ellfit_thresh={"min_shell_frac": 0.5,
                                             "min_angular_std": 0.7},
                              max_rad=1.5, verbose=False,
                              use_ransac=False, ransac_trials=50,
                              fit_iterations=2):
        '''
        Run find_bubbles on the specified scales.
        '''

        if scales is not None:
            self.scales = scales

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
                # Use + 1 sigma used to find local minima. Ensures the
                # edges that are found are real.
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

                for _ in range(fit_iterations):
                    coords = np.array(coords)
                    try_fit_ellipse = \
                        shell_frac >= ellfit_thresh["min_shell_frac"] and \
                        angular_std >= ellfit_thresh["min_angular_std"]

                    props, resid = \
                        fit_region(coords, initial_props=props,
                                   try_fit_ellipse=try_fit_ellipse,
                                   use_ransac=use_ransac,
                                   ransac_trials=ransac_trials,
                                   beam_pix=self.beam_pix, max_rad=max_rad,
                                   max_eccent=max_eccent,
                                   image_shape=self.array.shape,
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

                    if len(coords) < 3:
                        break
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

            # Append useful info onto the properties
            props = np.append(props, response_value)
            props = np.append(props, shell_frac)
            props = np.append(props, angular_std)
            props = np.append(props, resid)

            all_props.append(props)
            all_coords.append(np.array(coords))

        all_props = np.array(all_props)

        if not len(all_props) == 0:

            # First remove nearly duplicated regions. This stops much
            # smaller regions (corr <= 0.75) from dominating larger ones,
            # when they may only be a portion of the correct shape
            all_props, all_coords = \
                _prune_blobs(all_props, all_coords,
                             method="shell fraction",
                             min_corr=0.9, blob_merge=False)

            # Now look on smaller scales, and enable matching between smaller
            # regions embedded in a larger one. About 0.5 is appropriate since
            # a completely embedded smaller region will have 1/sqrt(3) for the
            # maximally eccentric shape allowed (e~3). This does make it
            # possible that a much larger region could be lost when it
            # shouldn't be. So long as the minimum cut used in the clustering
            # is ~0.5, these can still be clustered appropriately.
            all_props, all_coords = \
                _prune_blobs(all_props, all_coords,
                             method="shell fraction",
                             min_corr=overlap_frac)

            # Any highly overlapping regions should now be small regions
            # inside much larger ones. We're going to assume that the
            # remaining large regions are more important (good based on by-eye
            # evaluation). Keeping this at 0.65, since we only want to remove
            # very highly overlapping small regions. Note that this does set
            # an upper limit on how overlapped region may be. This is fairly
            # necessary though due to the shape ambiguities present in
            # assuming an elliptical shape.
            all_props, all_coords = \
                _prune_blobs(all_props, all_coords, overlap=0.65,
                             method='size')

            self._regions = \
                [Bubble2D(props, shell_coords=coords, channel=self.channel)
                 for props, coords in zip(all_props, all_coords)]
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

    def visualize_regions(self, show=True, edges=False, ax=None,
                          region_col='b', edge_col='g', log_scale=False):
        '''
        Show the regions optionally overlaid with the edges.
        '''

        if len(self.regions) == 0:
            Warning("No regions were found. Nothing to show.")
            return

        import matplotlib.pyplot as p

        if ax is None:
            ax = p.subplot(111)

        if log_scale:
            ax.imshow(np.log10(self.array), cmap='afmhot', origin='lower')
        else:
            ax.imshow(self.array, cmap='afmhot', origin='lower')

        for bub in self.regions:
            ax.add_patch(bub.as_patch(color=region_col, fill=False,
                                      linewidth=2))
            ax.plot(bub.x, bub.y, region_col + 'D')
            if edges:
                ax.plot(bub.shell_coords[:, 1], bub.shell_coords[:, 0],
                        edge_col + "o")

        p.xlim([0, self.array.shape[1]])
        p.ylim([0, self.array.shape[0]])

        if show:
            p.show()
        else:
            return ax
