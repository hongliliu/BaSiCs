
import numpy as np
from astropy.modeling.models import Ellipse2D
from astropy.stats import circmean
from spectral_cube.lower_dimensional_structures import LowerDimensionalObject
from spectral_cube import SpectralCube
from warnings import warn

from log import overlap_metric
from utils import floor_int, ceil_int, wrap_to_pi
from fan_pvslice import pv_wedge
from fit_models import fit_region


def no_specaxis_warning():
    warn("No spectral axis was provided.")


class BubbleNDBase(object):
    """
    Common properties between all cubes
    """

    @property
    def params(self):
        return np.array([self._y, self._x, self._major,
                         self._minor, self._pa, self._channel_center])

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
    def ra_extents(self):
        return self._ra_extents

    @property
    def dec(self):
        return self._dec

    @property
    def dec_extents(self):
        return self._dec_extents

    @property
    def channel_center(self):
        return self._channel_center

    @property
    def area(self):
        return np.pi * self.major * self.minor

    @property
    def perimeter(self):
        '''
        Estimate of the perimeter when major ~ minor.
        '''
        return 2 * np.pi * np.sqrt(0.5 * (self.major**2 + self.minor**2))

    @property
    def shell_coords(self):
        return self._shell_coords

    def as_patch(self, x_cent=None, y_cent=None, **kwargs):
        from matplotlib.patches import Ellipse
        y, x, rmaj, rmin, pa = self.params[:5]

        if y_cent is not None:
            y = y_cent
        if x_cent is not None:
            x = x_cent

        return Ellipse((x, y), width=2 * rmaj, height=2 * rmin,
                       angle=np.rad2deg(pa), **kwargs)

    def intensity_props(self, data):
        '''
        Return the mean and std for the elliptical region in the given data.
        '''

        if isinstance(data, LowerDimensionalObject):
            is_2D = True
        elif isinstance(data, SpectralCube):
            is_2D = False

        inner_ellipse = \
            Ellipse2D(True, self.x, self.y, max(3, self.major / 2.),
                      max(3, self.minor / 2.), self.pa)

        yy, xx = np.mgrid[:data.shape[-2], :data.shape[-1]]

        ellip_mask = inner_ellipse(xx, yy).astype(bool)

        if is_2D:
            return np.nanmean(data.value[ellip_mask]), \
                np.nanstd(data.value[ellip_mask])
        else:
            ellip_mask = np.tile(ellip_mask, (data.shape[0], 1, 1))
            return np.nanmean(data.with_mask(ellip_mask)), \
                np.nanstd(data.with_mask(ellip_mask))

    def set_wcs_extents(self, data):
        '''
        Set the spatial and/or spectral extents of the bubble.
        '''
        if isinstance(data, SpectralCube):

            self._ra = data.spatial_coordinate_map[0][self.center_pixel]
            self._dec = data.spatial_coordinate_map[1][self.center_pixel]

            y_extents, x_extents = self.find_spatial_extents()
            self._ra_extents = data.spatial_coordinate_map[0][np.c_[y_extents],
                                                              np.c_[x_extents]]
            self._dec_extents = \
                data.spatial_coordinate_map[1][np.c_[y_extents],
                                               np.c_[x_extents]]

            self._velocity_start = data.spectral_axis[self.channel_start]
            self._velocity_end = data.spectral_axis[self.channel_end]
            self._velocity_center = data.spectral_axis[self.channel_center]
            self._vel_width = np.abs(data.spectral_axis[1] -
                                     data.spectral_axis[0])

        elif isinstance(data, LowerDimensionalObject):
            # At some point, the 2D LDO will also have a
            # spatial_coordinate_map attribute
            raise NotImplementedError("")
        else:
            raise TypeError("data must be a SpectralCube or"
                            " LowerDimensionalObject.")


class Bubble2D(BubbleNDBase):
    """
    Class for candidate bubble portions from 2D planes.
    """
    def __init__(self, props, shell_coords=None, channel=None, data=None):
        super(Bubble2D, self).__init__()

        self._y = props[0]
        self._x = props[1]
        self._major = props[2]
        self._minor = props[3]
        self._pa = props[4]

        # > 6, some shell properties were included
        if len(props) > 6:
            self._peak_response = props[5]
            self._shell_fraction = props[6]
            self._angular_std = props[7]

        self._shell_coords = shell_coords

        # Requires finishing the WCS extents portion in set_wcs_extents
        # if data is not None:
        #     self.set_wcs_extents(data)

        # The last position is the velocity channel in the cube
        if channel is not None:
            self._channel_center = channel
        else:
            self._channel_center = 0

    def profile_lines(self, array, **kwargs):
        '''
        Calculate radial profile lines of the 2D bubbles.
        '''

        from basics.profile import radial_profiles

        return radial_profiles(array, self.params, **kwargs)

    @property
    def shell_fraction(self):
        '''
        Fraction of the region surrounded by a shell.
        '''
        return self._shell_fraction

    @property
    def angular_std(self):
        '''
        The angular standard deviation of the shell positions.
        '''
        return self._angular_std

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
            y_range = ceil_int((bbox[0][1] - bbox[0][0]))
            x_range = ceil_int((bbox[1][1] - bbox[1][0]))

            yy, xx = np.mgrid[-int(y_range / 2): int(y_range / 2) + 1,
                              -int(x_range / 2): int(x_range / 2) + 1]

        else:
            yy, xx = np.mgrid[:shape[0], :shape[1]]

        return self.as_ellipse(zero_center=zero_center)(xx, yy).astype(bool)

    def find_spatial_extents(self):
        '''
        Get the pixel extents of the region, based on the minimal bounding box
        of the ellipse.
        '''
        return self.as_ellipse(zero_center=False).bounding_box

    def slice_to_region(self, array, pad=None):
        '''
        Return the region defined by the bounding box in the given array.
        '''

        if pad is None:
            pad = 0

        bbox = self.find_spatial_extents()

        return array[floor_int(bbox[0][0]) - pad:
                     ceil_int(bbox[0][1]) + pad + 1,
                     floor_int(bbox[1][0]) - pad:
                     ceil_int(bbox[1][1]) + pad + 1]

    def overlap_with(self, other_bubble2D):
        '''
        Return the overlap with another bubble.
        '''

        return overlap_metric(self.params, other_bubble2D.params)

    def __repr__(self):
        s = "2D Bubble at: ({0:6f}, {1:6f})\n".format(self.y, self.x)
        if self.major == self.minor:
            s += "Radius: {0:6f} \n".format(self.major)
        else:
            s += "Major radius: {0:6f} \n".format(self.major)
            s += "Minor radius: {0:6f} \n".format(self.minor)
            s += "Position Angle: {0:6f} \n".format(self.pa)

        if self.shell_fraction is not None:
            s += "Shell fraction: {0:6f} \n".format(self.shell_fraction)

        return s


class Bubble3D(BubbleNDBase):
    """
    3D Bubbles.

    Parameters
    ----------
    cube : SpectralCube
        Uses the cube to find the spatial and spectral extents of the bubble.
    """
    def __init__(self, props, cube=None, twoD_regions=None):
        super(Bubble3D, self).__init__()

        self._y = props[0]
        self._x = props[1]
        self._major = props[2]
        self._minor = props[3]
        self._pa = props[4]
        self._channel_center = props[5]
        self._channel_start = props[6]
        self._channel_end = props[7]

        self.twoD_regions = twoD_regions

        if cube is not None:
            self.set_wcs_extents(cube)

    @staticmethod
    def from_2D_regions(twod_region_list, refit=True,
                        cube=None, **fit_kwargs):
        '''
        Create a 3D regions from a collection of 2D regions.
        '''

        # Extract the 2D properties
        twoD_properties = \
            np.array([bub2D.params for bub2D in twod_region_list])

        # Sort by channel
        twod_region_list = \
            [twod_region_list[i] for i in twoD_properties[:, -1].argsort()]

        all_coords = []
        for reg in twod_region_list:
            chan_coord = reg.channel_center * \
                np.ones((reg.shell_coords.shape[0], 1))
            all_coords.append(np.hstack([chan_coord, reg.shell_coords]))

        all_coords = np.vstack(all_coords)

        if refit:
            props, resid = fit_region(all_coords[:, 1:], **fit_kwargs)

        else:
            props = np.array([np.median(twoD_properties[:, 0]),
                              np.median(twoD_properties[:, 1]),
                              np.median(twoD_properties[:, 2]),
                              np.median(twoD_properties[:, 3]),
                              wrap_to_pi(circmean(twoD_properties[:, 4]))])

        props = np.append(props,
                          [int(round(np.median(twoD_properties[:, 5]))),
                           int(twoD_properties[:, 5].min()),
                           int(twoD_properties[:, 5].max())])

        self = Bubble3D(props, cube=cube, twoD_regions=twod_region_list)

        self._shell_coords = all_coords

        return self

    @property
    def twoD_regions(self):
        return self._twoD_regions

    @twoD_regions.setter
    def twoD_regions(self, input_list):
        if input_list is not None:
            for reg in input_list:
                if isinstance(reg, Bubble2D):
                    continue

                raise TypeError("twoD_regions must be a list of Bubble2D"
                                " objects")

        self._twoD_regions = input_list

    @property
    def has_2D_regions(self):
        return True if self.twoD_regions is not None else False

    @property
    def velocity_start(self):
        if self._velocity_start is not None:
            return self._velocity_start
        no_specaxis_warning()

    @property
    def velocity_end(self):
        if self._velocity_end is not None:
            return self._velocity_end
        no_specaxis_warning()

    @property
    def velocity_center(self):
        if self._velocity_center is not None:
            return self._velocity_center
        no_specaxis_warning()

    @property
    def velocity_width(self):
        if self._vel_width is not None:
            return self.channel_width * self._vel_width
        no_specaxis_warning()

    @property
    def channel_start(self):
        return self._channel_start

    @property
    def channel_end(self):
        return self._channel_end

    @property
    def channel_width(self):
        return self.channel_end - self.channel_start + 1

    @property
    def bubble_type(self):
        return self._bubble_type

    def _chan_iter(self):
        return xrange(int(self.channel_start), int(self.channel_end) + 1)

    def _twoD_region_iter(self):
        for region in self.twoD_regions:
            yield region

    def twoD_region_params(self):
        return np.array([region.params for region in self._twoD_region_iter()])

    def find_spatial_extents(self, zero_center=True):
        '''
        Find the maximum spatial extents.
        '''

        if not self.has_2D_regions:
            raise NotImplementedError("")

        bboxes = np.empty((4, len(self.twoD_regions)), dtype=np.int)

        for i, region in enumerate(self._twoD_region_iter()):
            bbox = region.as_ellipse(zero_center=zero_center).bounding_box
            bboxes[:2, i] = bbox[0]
            bboxes[2:, i] = bbox[1]

        return [[floor_int(np.min(bboxes[0])), ceil_int(np.max(bboxes[1]))],
                [floor_int(np.min(bboxes[2])), ceil_int(np.max(bboxes[3]))]]

    def extract_pv_slice(self, cube, width=None, use_subcube=True, **kwargs):
        '''
        Return a PV Slice. Defaults to across the entire bubble.
        '''

        try:
            import pvextractor
        except ImportError:
            raise ImportError("pvextractor must be installed to extract "
                              " PV slices.")

        if "spatial_pad" in kwargs:
            spatial_pad = kwargs["spatial_pad"]
        else:
            spatial_pad = 0

        # Define end points along the major axis
        max_dist = 2 * float(self.major + spatial_pad)

        if width is None:
            width = 1  # self.minor

        if use_subcube:

            subcube = self.return_cube_region(cube, **kwargs)

            sub_center = (floor_int(subcube.shape[1] / 2.),
                          floor_int(subcube.shape[2] / 2.))

            return pv_wedge(subcube, sub_center, max_dist, 0.0, np.pi,
                            width=width)
        else:
            return pv_wedge(cube, self.center, max_dist, 0.0, np.pi,
                            width=width)

    def as_pv_patch(self, x_cent=None, chan_cent=None, **kwargs):
        '''
        Return a PV slice. Aligns the direction along the major axis.
        '''
        from matplotlib.patches import Ellipse

        if x_cent is None:
            x_cent = self.major

        if chan_cent is None:
            chan_cent = self.channel_center

        return Ellipse((x_cent, chan_cent),
                       width=2 * self.major,
                       height=self.channel_width,
                       angle=0.0, **kwargs)

    def as_mask(self, spatial_shape, zero_center=False):
        '''
        Return an elliptical mask.
        '''

        if len(spatial_shape) != 2:
            raise ValueError("spatial_shape must have a length of 2.")

        if not self.has_2D_regions:
            raise NotImplementedError("")

        ellip_mask = np.zeros((len(self.twoD_regions),) + spatial_shape,
                              dtype=bool)

        for i, region in enumerate(self._twoD_region_iter()):
            ellip_mask[i] = \
                region.as_mask(shape=spatial_shape,
                               zero_center=zero_center)

        return ellip_mask

    def as_shell_mask(self, cube):
        pass

    def slice_to_bubble(self, cube, spatial_pad=0, spec_pad=0):
        '''
        Return the minimum subcube about the bubble.
        '''

        # y, x  extents
        extents = self.find_spatial_extents(zero_center=False)

        spec_slice = slice(max(0, self.channel_start - spec_pad),
                           min(cube.shape[0],
                               self.channel_end + 1 + spec_pad), 1)

        y_slice = slice(max(0, extents[0][0] - spatial_pad),
                        min(cube.shape[1], extents[0][1] + spatial_pad), 1)
        x_slice = slice(max(0, extents[1][0] - spatial_pad),
                        min(cube.shape[1], extents[1][1] + spatial_pad), 1)

        subcube = cube[spec_slice, y_slice, x_slice]

        return subcube

    def return_moment0(self, cube, **kwargs):
        '''
        Return the moment 0 of the sub-cube created by `return_cube_region`.
        '''

        return self.slice_to_bubble(cube, **kwargs).moment0()

    def visualize_shell(self):
        '''
        Make a 3D point plot of the shell coordinates.
        '''

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as p

        fig = p.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.shell_coords[:, 2], self.shell_coords[:, 1],
                   self.shell_coords[:, 0])

        # Flush this out with proper labels, etc...
        ax.set_xlabel("x")
        ax.set_xlabel("y")
        ax.set_xlabel("v")

        p.show()

    def visualize(self, cube, return_plot=False, use_aplpy=True):
        '''
        Visualize the bubble within the cube.
        '''

        moment0 = self.return_moment0(cube, spatial_pad=10, spec_pad=5)

        pvslice = \
            self.extract_pv_slice(cube, spatial_pad=10, spec_pad=5)

        import matplotlib.pyplot as p
        fig = p.figure()

        if use_aplpy:
            try:
                import aplpy
                mom0_fig = aplpy.FITSFigure(moment0.hdu, figure=fig,
                                            subplot=(1, 2, 1))
                mom0_fig.show_grayscale()
                mom0_fig.add_colorbar()

                pv_fig = aplpy.FITSFigure(pvslice, figure=fig,
                                          subplot=(1, 2, 2))
                pv_fig.show_grayscale()
                pv_fig.add_colorbar()

            except ImportError:
                Warning("aplpy not installed. Reverting to matplotlib.")
                use_aplpy = False

        if not use_aplpy:
            ax1 = fig.add_subplot(121)
            im1 = ax1.imshow(moment0.value, origin='lower', cmap='gray',
                             interpolation='nearest')
            fig.colorbar(im1, ax=ax1)

            c = self.as_patch(x_cent=floor_int(moment0.shape[1] / 2.),
                              y_cent=floor_int(moment0.shape[0] / 2.),
                              fill=False, color='r', linewidth=2)
            ax1.add_patch(c)
            # ax1.arrow(path_ends[0][0], path_ends[0][1],
            #           path_ends[1][0] - path_ends[0][0],
            #           path_ends[1][1] - path_ends[0][1],
            #           fc='r', ec='r')

            ax2 = fig.add_subplot(122)
            im2 = ax2.imshow(pvslice.data, origin='lower', cmap='gray',
                             interpolation='nearest')
            fig.colorbar(im2, ax=ax2)

            c = self.as_pv_patch(fill=False, color='r', linewidth=2,
                                 chan_cent=pvslice.data.shape[0] / 2,
                                 x_cent=pvslice.data.shape[1] / 2)
            ax2.add_patch(c)

        if return_plot:
            return fig

        p.show()

    def __repr__(self):
        s = "3D Bubble at: ({0:6f}, {1:6f}," \
            "{2:6f})\n".format(self.channel_center, self.y, self.x)
        if self.major == self.minor:
            s += "Radius: {0:6f} \n".format(self.major)
        else:
            s += "Major radius: {0:6f} \n".format(self.major)
            s += "Minor radius: {0:6f} \n".format(self.minor)
            s += "Position Angle: {0:6f} \n".format(self.pa)

        s += "Channel width: {0:6f} \n".format(self.channel_width)
        # s += "Spectral width: {0.6f} \n".format(self.velocity_width)

        # if self.shell_fraction is not None:
        #     s += "Shell fraction: {0:6f} \n".format(self.shell_fraction)

        return s
