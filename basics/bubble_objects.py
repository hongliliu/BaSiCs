
import numpy as np
from astropy.modeling.models import Ellipse2D
from astropy.nddata.utils import extract_array, add_array
from scipy import ndimage as nd
from itertools import izip
from spectral_cube.lower_dimensional_structures import LowerDimensionalObject
from spectral_cube.base_class import BaseNDClass

from log import overlap_metric
from utils import consec_split, find_nearest, floor_int, ceil_int
from profile import _line_profile_coordinates

eight_conn = np.ones((3, 3))


class BubbleNDBase(BaseNDClass):
    """
    Common properties between all cubes
    """

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def center_pixel(self):
        return (floor_int(self.y), floor_int(self.x))

    @property
    def pa(self):
        return self._pa

    @property
    def major(self):
        return self._major

    @property
    def minor(self):
        return self._minor

    @property
    def ra(self):
        return self._ra

    @property
    def dec(self):
        return self._dec


class Bubble2D(BubbleNDBase):
    """
    Class for candidate bubble portions from 2D planes.
    """
    def __init__(self, props, wcs=None):
        super(Bubble2D, self).__init__()

        self._y = props[0]
        self._x = props[1]
        self._major = props[2]
        self._minor = props[3]
        self._pa = props[4]

        # The last position, if given, is the velocity channel in the cube
        try:
            self._channel = props[5]
        except IndexError:
            self._channel = None

        self._wcs = None

    @property
    def params(self):
        if self.channel is not None:
            return np.array([self._y, self._x, self._major,
                             self._minor, self._pa, self._channel])

        return np.array([self._y, self._x, self._major,
                         self._minor, self._pa])

    @property
    def area(self):
        return np.pi * self.major * self.minor

    @property
    def perimeter(self):
        '''
        Estimate of the perimeter when major ~ minor.
        '''
        return 2 * np.pi * np.sqrt(0.5*(self.major**2 + self.minor**2))

    @property
    def channel(self):
        if self._channel is None:
            Warning("Bubble2D not instantiated with a velocity channel.")

        return self._channel

    def profile_lines(self, array, **kwargs):
        '''
        Calculate radial profile lines of the 2D bubbles.
        '''

        from basics.profile import radial_profiles

        return radial_profiles(array, self.params, **kwargs)

    def find_shell_fraction(self, array, value_thresh=0.0,
                            grad_thresh=1, **kwargs):
        '''
        Find the fraction of the bubble edge associated with a shell.
        '''

        shell_frac = 0

        # Set the number of theta to be ~ the perimeter.
        ntheta = 1.5 * np.ceil(self.perimeter).astype(int)

        for dist, prof in self.profile_lines(array, ntheta=ntheta, **kwargs):

            above_thresh = prof >= value_thresh

            nabove = above_thresh.sum()

            if nabove < max(2, 0.05*len(above_thresh)):
                continue

            shell_frac += 1

        self._shell_fraction = float(shell_frac) / float(ntheta)

    @property
    def shell_fraction(self):
        return self._shell_fraction

    def as_ellipse(self, zero_center=True):
        '''
        Returns an Ellipse2D model.

        Parameters
        ----------
        zero_center : bool, optional
            If enabled, returns a model with a centre at (0, 0). Otherwise the
            centre is at the pixel position in the array.
        '''
        if zero_center:
            return Ellipse2D(True, 0.0, 0.0, self.major, self.minor,
                             self.pa)
        return Ellipse2D(True, self.x, self.y, self.major, self.minor,
                         self.pa)

    def as_mask(self, shape=None, zero_center=False):
        '''
        Return a boolean mask of the 2D region.
        '''

        # Returns the bbox shape. Forces zero_center to be True
        if shape is None:
            zero_center = True
            bbox = self.as_ellipse(zero_center=False).bounding_box
            y_range = np.ceil((bbox[0][1] - bbox[0][0])).astype(int)
            x_range = np.ceil((bbox[1][1] - bbox[1][0])).astype(int)

            yy, xx = np.mgrid[-int(y_range / 2): int(y_range / 2) + 1,
                              -int(x_range / 2): int(x_range / 2) + 1]

        else:
            yy, xx = np.mgrid[:shape[0], :shape[1]]

        return self.as_ellipse(zero_center=zero_center)(xx, yy).astype(bool)

    def return_array_region(self, array, pad=None):
        '''
        Return the region defined by the bounding box in the given array.
        '''

        if pad is None:
            pad = 0

        bbox = self.as_ellipse(zero_center=False).bounding_box

        return array[np.floor(bbox[0][0]).astype(int)-pad:
                     np.ceil(bbox[0][1]).astype(int)+pad+1,
                     np.floor(bbox[1][0]).astype(int)-pad:
                     np.ceil(bbox[1][1]).astype(int)+pad+1]

    def intensity_props(self, array):
        '''
        Return the mean and std for the elliptical region in the given array.
        '''

        if isinstance(array, LowerDimensionalObject):
            array = array.value.copy()

        inner_ellipse = \
            Ellipse2D(True, self.x, self.y, max(3, self.major/2.),
                      max(3, self.minor/2.), self.pa)
        yy, xx = np.mgrid[:array.shape[0], :array.shape[1]]

        ellip_mask = inner_ellipse(xx, yy).astype(bool)

        masked_array = array.copy()
        masked_array[~ellip_mask] = np.NaN

        return np.nanmean(masked_array), np.nanstd(masked_array)

    def find_shape(self, array, return_array='full', max_extent=1.0,
                   nsig_thresh=1, value_thresh=None, min_radius_frac=0.5,
                   radius=None, **kwargs):
        '''
        Expand/contract to match the contours in the data.

        Parameters
        ----------
        array : 2D numpy.ndarray or spectral_cube.LowerDimensionalObject
            Data used to define the region boundaries.
        return_array : 'full', 'bbox', or 'padded', optional
            Choose the shape of the mask to be returned. 'full' returns
            the mask within the full array. 'bbox' cuts to the boundaries of
            the ellipse. 'padded' returns a padded version of 'bbox' that
            depends on the choice of max_extent.
        max_extent : float, optional
            Multiplied by the major radius to set how far should be searched
            when searching for the boundary.
        nsig_thresh : float, optional
            Number of times sigma above the mean to set the boundary intensity
            requirement. This is used whenever the local background is higher
            than the given `value_thresh`.
        value_thresh : float, optional
            When given, sets the minimum intensity for defining a bubble edge.
            The natural choice is a few times the noise level in the cube.
        min_radius_frac : float, optional
            Sets a minimum distance to search for the bubble boundary. Defaults
            to 1/2 of the major radius.
        radius : float, optional
            Give an optional radius to use instead of the major radius defined
            for the bubble.
        kwargs : passed to profile.profile_line.

        Returns
        -------
        extent_mask : np.ndarray
            Boolean array with the region mask.
        '''

        mean, std = self.intensity_props(array)
        background_thresh = mean + nsig_thresh * std

        # Define a suitable background based on the intensity within the
        # elliptical region
        if value_thresh is None:
            value_thresh = background_thresh
        else:
            # If value_thresh is higher use it. Otherwise use the bkg.
            if value_thresh < background_thresh:
                value_thresh = background_thresh

        # Set the number of theta to be ~ the perimeter.
        ntheta = 1.5 * np.ceil(self.perimeter).astype(int)

        # Use the ellipse model to define a bounding box for the mask.
        bbox = self.as_ellipse(zero_center=True).bounding_box

        y_range = np.ceil((max_extent * 2) *
                          (bbox[0][1] - bbox[0][0])).astype(int)
        x_range = np.ceil((max_extent * 2) *
                          (bbox[1][1] - bbox[1][0])).astype(int)

        extent_mask = np.zeros((y_range, x_range), dtype=bool)

        yy, xx = np.mgrid[-int(y_range / 2): int(y_range / 2) + 1,
                          -int(x_range / 2): int(x_range / 2) + 1]

        dist_arr = np.sqrt(yy**2 + xx**2)

        centre = [arr[0] for arr in np.where(dist_arr == 0.0)]

        for dist, prof, end in self.profile_lines(array,
                                                  extend_factor=max_extent,
                                                  append_end=True,
                                                  ntheta=ntheta, **kwargs):

            new_end = (max_extent * (end[0] - self.y),
                       max_extent * (end[1] - self.x))

            max_dist = np.abs(dist.max())

            line_posns = []
            coords = [np.floor(coord).astype(int)+cent for coord, cent in
                      zip(_line_profile_coordinates((0, 0), new_end), centre)]
            for coord in coords:
                line_posns.append(coord[dist_arr[coords] >=
                                  min_radius_frac*self.minor])

            prof = prof[dist >= min_radius_frac * self.minor]
            dist = dist[dist >= min_radius_frac * self.minor]

            above_thresh = np.where(prof >= value_thresh)[0]

            # This angle does not coincide with a shell.
            # We fill in a pixel at the major radius.
            if above_thresh.size == 0:
                nearest_idx = find_nearest(dist_arr[line_posns],
                                           max_dist).astype(int)
            else:
                # Pick the end point on the first segment with > 2 pixels
                # above the threshold.
                segments = consec_split(above_thresh)

                for seg in segments:
                    if seg.size < max(3, int(0.1*above_thresh.size)):
                        continue

                    # Take the first position and use it to define the edge
                    dist_val = dist[seg[0]]

                    nearest_idx = \
                        find_nearest(dist_arr[line_posns],
                                     dist_val).astype(int)
                    break
                else:
                    # If no segments larger than 2 pixels are found, default
                    # to the major radius
                    nearest_idx = \
                        find_nearest(dist_arr[line_posns],
                                     max_dist).astype(int)

            end_posn = tuple([posn[:nearest_idx+1] for posn in line_posns])

            extent_mask[end_posn] = True

        # Fill holes
        extent_mask = nd.binary_fill_holes(extent_mask)
        extent_mask = nd.binary_closing(extent_mask, eight_conn)
        extent_mask = nd.binary_opening(extent_mask, eight_conn)

        if return_array is "bbox":
            bbox_shape = \
                (np.ceil(bbox[0][1]).astype(int) -
                 np.floor(bbox[0][0]).astype(int),
                 np.ceil(bbox[1][1]).astype(int) -
                 np.floor(bbox[1][0]).astype(int))

            return extract_array(extent_mask, bbox_shape, centre)
        elif return_array is "full":
            return add_array(np.zeros_like(array.value, dtype=bool),
                             extent_mask, self.center_pixel)
        elif return_array is "padded":
            return extent_mask
        else:
            raise ValueError("return_array must be 'bbox', 'full', or"
                             " 'padded'.")

    def overlap_with(self, other_bubble2D):
        '''
        Return the overlap with another bubble.
        '''

        return overlap_metric(self.params, other_bubble2D.params,
                              includes_channel=False)

    def as_patch(self, **kwargs):
        from matplotlib.patches import Ellipse
        y, x, rmaj, rmin, pa = self.params[:5]
        return Ellipse((x, y), width=2*rmaj, height=2*rmin,
                       angle=np.rad2deg(pa), **kwargs)


class Bubble3D(BubbleNDBase):
    """
    3D Bubbles.
    """
    def __init__(self, props, wcs=None):
        super(Bubble3D, self).__init__()

        self._y = props[0]
        self._x = props[1]
        self._major = props[2]
        self._minor = props[3]
        self._pa = props[4]
        self._velocity_center = props[5]
        self._velocity_start = props[6]
        self._velocity_end = props[7]

        self._twoD_objects = None

        self._wcs = wcs

    @staticmethod
    def from_2D_regions(twod_region_list, wcs=None):
        '''
        Create a 3D regions from a collection of 2D regions.
        '''

        # Extract the 2D properties
        twoD_properties = \
            np.array([bub2D.params for bub2D in twod_region_list])

        # Sort by channel
        twod_region_list = \
            [twod_region_list[i] for i in twoD_properties[:, 0].argsort()]

        props = [twoD_properties[:, 0].mean(), twoD_properties[:, 1].mean(),
                 twoD_properties[:, 2].max(), twoD_properties[:, 3].max(),
                 twoD_properties[:, 4].mean(),
                 int(round(np.median(twoD_properties[:, 5]))),
                 int(twoD_properties[:, 5].min()),
                 int(twoD_properties[:, 5].max())]

        self = Bubble3D(props, wcs=wcs)

        self._twoD_objects = twod_region_list

        return self

    @property
    def twoD_objects(self):
        return self._twoD_objects

    @property
    def has_2D_regions(self):
        return True if self.twoD_objects is not None else False

    @property
    def velocity_start(self):
        return self._velocity_start

    @property
    def velocity_end(self):
        return self._velocity_end

    @property
    def velocity_center(self):
        return self._velocity_center

    @property
    def velocity_width(self):
        return self.velocity_end - self.velocity_start + 1

    @property
    def bubble_type(self):
        return self._bubble_type

    def _chan_iter(self):
        return xrange(int(self.velocity_start), int(self.velocity_end)+1)

    def _twoD_region_iter(self):
        for region in self.twoD_objects:
            yield region

    def twoD_region_params(self):
        return np.array([region.params for region in self._twoD_region_iter()])

    def find_spatial_extents(self, zero_center=True):
        '''
        Find the maximum spatial extents.
        '''

        if not self.has_2D_regions:
            raise NotImplementedError("")

        bboxes = np.empty((4, self.velocity_width), dtype=np.int)

        for i, region in enumerate(self._twoD_region_iter()):
            bbox = region.as_ellipse(zero_center=zero_center).bounding_box
            bboxes[:2, i] = bbox[0]
            bboxes[2:, i] = bbox[1]

        return (floor_int(np.min(bboxes[0])), ceil_int(np.max(bboxes[1])),
                floor_int(np.min(bboxes[2])), ceil_int(np.max(bboxes[3])))

    def extract_pv_slice(self, cube, width=None, use_subcube=True, **kwargs):
        '''
        Return a PV Slice. Defaults to across the entire bubble.
        '''

        try:
            from pvextractor import Path, extract_pv_slice
        except ImportError:
            raise ImportError("pvextractor must be installed to extract "
                              " PV slices.")

        if width is None:
            width = 2*self.minor

        # Set width to be twice minor radius

        if use_subcube:

            subcube = self.return_cube_region(cube, **kwargs)

            y_center = floor_int(subcube.shape[1]/2.)
            x_center = floor_int(subcube.shape[2]/2.)

            # Define end points along the major axis
            high = (y_center + self.major*np.sin(self.pa),
                    x_center + self.major*np.cos(self.pa))
            low = (y_center - self.major*np.sin(self.pa),
                   x_center - self.major*np.cos(self.pa))

            return extract_pv_slice(subcube,
                                    Path([low, high], width=width))
        else:
            high = (self.y + self.major*np.sin(self.pa),
                    self.x + self.major*np.cos(self.pa))
            low = (self.y - self.major*np.sin(self.pa),
                   self.x - self.major*np.cos(self.pa))

            return extract_pv_slice(cube, Path([low, high], width=width))

    def as_mask(self, spatial_shape, zero_center=False):
        '''
        Return an elliptical mask.
        '''

        if len(spatial_shape) != 2:
            raise ValueError("spatial_shape must have a length of 2.")

        if not self.has_2D_regions:
            raise NotImplementedError("")

        ellip_mask = np.zeros((len(self.twoD_objects),) + spatial_shape,
                              dtype=bool)

        for i, region in enumerate(self._twoD_region_iter()):
            ellip_mask[i] = \
                region.as_mask(shape=spatial_shape,
                               zero_center=zero_center)

        return ellip_mask

    def as_extent_mask(self, cube, **kwargs):
        '''
        Run Bubble2D.find_shape to get the extend mask in each channel.
        '''

        if not self.has_2D_regions:
            raise NotImplementedError("")

        for i, (chan, region) in enumerate(izip(self._chan_iter(),
                                                self._twoD_region_iter())):
            extent_2d_mask = region.find_shape(cube[chan], **kwargs)

            if i == 0:
                extent_mask = extent_2d_mask[np.newaxis, :, :]
            else:
                extent_mask = np.append(extent_mask,
                                        extent_2d_mask[np.newaxis, :, :],
                                        axis=0)

        return extent_mask

    def as_shell_mask(self, cube):
        pass

    def return_cube_region(self, cube, spatial_pad=0, spec_pad=0):
        '''
        Return the minimum subcube about the bubble.
        '''

        # y, x  extents
        extents = self.find_spatial_extents(zero_center=False)

        spec_slice = slice(max(0, self.velocity_start-spec_pad),
                           min(cube.shape[0],
                               self.velocity_end+1+spec_pad), 1)

        y_slice = slice(max(0, extents[0]-spatial_pad),
                        min(cube.shape[1], extents[1]+spatial_pad), 1)
        x_slice = slice(max(0, extents[2]-spatial_pad),
                        min(cube.shape[1], extents[3]+spatial_pad), 1)

        subcube = cube[spec_slice, y_slice, x_slice]

        return subcube

    def return_moment0(self, cube, **kwargs):
        '''
        Return the moment 0 of the sub-cube created by `return_cube_region`.
        '''

        return self.return_cube_region(cube, **kwargs).moment0()

    def visualize(self, cube, return_plot=False, use_aplpy=True):
        '''
        Visualize the bubble within the cube.
        '''

        moment0 = self.return_moment0(cube, spatial_pad=10, spec_pad=5)

        pvslice = self.extract_pv_slice(cube, spatial_pad=10, spec_pad=5)

        import matplotlib.pyplot as p
        fig = p.figure()

        if use_aplpy:
            try:
                import aplpy
                mom0_fig = aplpy.FITSFigure(moment0.hdu, figure=fig,
                                            subplot=(1, 2, 1))
                mom0_fig.show_grayscale()
                mom0_fig.add_colorbar()

                pv_fig = aplpy.FITSFigure(pvslice.data, figure=fig,
                                          subplot=(1, 2, 2))
                pv_fig.show_grayscale()
                pv_fig.add_colorbar()

            except ImportError:
                Warning("aplpy not installed. Reverting to matplotlib.")
                use_aplpy = False

        if not use_aplpy:
            ax1 = fig.add_subplot(121)
            ax1.imshow(moment0, origin='lower', cmap='gray')
            ax1.colorbar()

            ax2 = fig.add_subplot(121)
            ax2.imshow(pvslice.data, origin='lower', cmap='gray')
            ax2.colorbar()

        if return_plot:
            return fig

        p.show()
