
import numpy as np
from astropy.modeling.models import Ellipse2D
from astropy.stats import circmean
from astropy import units as u
from spectral_cube.lower_dimensional_structures import LowerDimensionalObject
from spectral_cube import SpectralCube
from warnings import warn

from log import overlap_metric
from utils import floor_int, ceil_int, wrap_to_pi
from fan_pvslice import pv_wedge
from fit_models import fit_region


def no_wcs_warning():
    warn("No data product with an attached WCS was provided.")


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
    def diameter(self):
        return 2 * np.sqrt(self.major * self.minor)

    @property
    def ra(self):
        if self._ra is not None:
            return self._ra
        no_wcs_warning()

    @property
    def ra_extents(self):
        if self._ra_extents is not None:
            return self._ra_extents
        no_wcs_warning()

    @property
    def dec(self):
        if self._dec is not None:
            return self._dec
        no_wcs_warning()

    @property
    def dec_extents(self):
        if self._dec_extents is not None:
            return self._dec_extents
        no_wcs_warning()

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

    @property
    def shell_fraction(self):
        '''
        Fraction of the region surrounded by a shell.
        '''
        return self._shell_fraction

    @property
    def is_closed(self):
        '''
        Determine whether the bubble is closed or partial. Closed is defined
        by having a shell fraction >0.9, which follows the definition of
        Deul & den Hartog (1990).
        '''
        # Bubble3D only has this set if 2D regions were given
        if not hasattr(self, "_shell_fraction"):
            raise Warning("Bubble3D must be created from Bubble2D objects for"
                          " shell_fraction to be defined.")

        return True if self.shell_fraction >= 0.9 else False

    def overlap_with(self, other_bubble):
        '''
        Return the overlap with another bubble.
        '''

        return overlap_metric(self.params, other_bubble.params)

    def as_patch(self, x_cent=None, y_cent=None, **kwargs):
        from matplotlib.patches import Ellipse
        y, x, rmaj, rmin, pa = self.params[:5]

        if y_cent is not None:
            y = y_cent
        if x_cent is not None:
            x = x_cent

        return Ellipse((x, y), width=2 * rmaj, height=2 * rmin,
                       angle=np.rad2deg(pa), **kwargs)

    def intensity_props(self, data, area_factor=1.):
        '''
        Return the mean and std for the elliptical region in the given data.
        '''

        if isinstance(data, LowerDimensionalObject):
            is_2D = True
        elif isinstance(data, SpectralCube):
            is_2D = False

        # Assume the shape is evenly increased/decreased by area_facotr
        major = max(3, np.sqrt(area_factor) * self.major)
        minor = max(3, np.sqrt(area_factor) * self.minor)

        inner_ellipse = \
            Ellipse2D(True, self.x, self.y, major, minor, self.pa)

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

    def set_shell_properties(self, data, mask, **shell_kwargs):
        '''
        Get the properties of the shell
        '''

        # Requires masking for 2D objects in SpectralCube to be ready
        if isinstance(self, Bubble2D):
            raise NotImplementedError("")

        spectral_extent = False
        if isinstance(self, Bubble3D):
            spectral_extent = True

        if "spectral_extent" in shell_kwargs:
            warn("spectral_extent is automatically set. Ignoring input.")
            del shell_kwargs["spectral_extent"]

        shell_mask = self.as_shell_mask(mask=mask,
                                        spectral_extent=spectral_extent,
                                        **shell_kwargs)

        # Apply the mask, then cut to its extents
        shell_cube = data.with_mask(shell_mask).minimal_subcube()
        # Once 2D is supported, need extra if/else for minimal_sub-something

        mom0 = shell_cube.moment0()
        # Can't just use mom0.mean() right now until this issue is dealt with:
        # https://github.com/radio-astro-tools/spectral-cube/issues/279
        self._avg_shell_flux_density = np.nanmean(mom0)
        self._total_shell_flux_density = np.nansum(mom0)

        # In 3D, set the velocity properties
        if isinstance(self, Bubble3D):
            self._shell_velocity_median = np.nanmedian(shell_cube.moment1())
            self._shell_velocity_disp = np.nanmedian(shell_cube.moment2())

    @property
    def avg_shell_flux_density(self):
        return self._avg_shell_flux_density

    @property
    def total_shell_flux_density(self):
        return self._total_shell_flux_density

    def as_ellipse(self, zero_center=True, extend_factor=1):
        '''
        Returns an Ellipse2D model.

        Parameters
        ----------
        zero_center : bool, optional
            If enabled, returns a model with a centre at (0, 0). Otherwise the
            centre is at the pixel position in the array.
        '''

        major = extend_factor * self.major
        minor = extend_factor * self.minor

        if zero_center:
            return Ellipse2D(True, 0.0, 0.0, major, minor,
                             self.pa)
        return Ellipse2D(True, self.x, self.y, major, minor,
                         self.pa)

    def as_mask(self, mask=None, shape=None, zero_center=False,
                spectral_extent=False, use_twoD_regions=False):
        '''
        Return a boolean mask of the 2D region.

        Parameters
        ----------
        shape : tuple, optional
            The shape of the mask to output. When returning a 3D mask, the
            spectral size *MUST* match the spectral size of the original cube!
            When no shape is given, a mask with the minimal shape of the
            bubble is returned.

        '''

        # This situation seems ill-defined unless the mask shape is assumed.
        # Force to the mask shape in this case.
        if shape is None and mask is not None:
            shape = mask.shape

        if use_twoD_regions and shape is None and mask is None:
            raise NotImplementedError("Issues with defining a single grid for"
                                      " each local region.")

        # Returns the bbox shape. Forces zero_center to be True
        if shape is None:
            zero_center = True
            bbox = self.as_ellipse(zero_center=False).bounding_box
            y_range = ceil_int((bbox[0][1] - bbox[0][0]))
            x_range = ceil_int((bbox[1][1] - bbox[1][0]))

            yy, xx = np.mgrid[-int(y_range / 2): int(y_range / 2) + 1,
                              -int(x_range / 2): int(x_range / 2) + 1]

        else:
            if len(shape) == 2:
                yshape, xshape = shape
            elif len(shape) == 3:
                yshape, xshape = shape[1:]
            else:
                raise TypeError("shape must be for 2D or 3D.")

            yy, xx = np.mgrid[:yshape, :xshape]

        twoD_mask = \
            self.as_ellipse(zero_center=zero_center)(xx, yy).astype(np.bool)

        # Just return the 2D footprint
        if not spectral_extent:
            region_mask = twoD_mask
        elif spectral_extent and isinstance(self, Bubble2D):
            raise TypeError("Can only use spectral_extent for Bubble3D"
                            " objects.")
        else:

            if shape is None:
                nchans = self.channel_width
                start = 0
                end = self.channel_width + 1
                yshape = yy.shape[0]
                xshape = yy.shape[1]
            else:
                if len(shape) != 3:
                    raise TypeError("A 3D shape must be given when returning"
                                    " a 3D mask.")
                nchans = shape[0]
                start = self.channel_start
                end = self.channel_end + 1
                yshape = shape[1]
                xshape = shape[2]

            if use_twoD_regions:
                region_mask = \
                    np.zeros((nchans, yshape, xshape),
                             dtype=np.bool)

                chans = np.arange(start, end).astype(int)
                for i, region in zip(chans, self._twoD_region_iter()):
                    region_mask[i] = \
                        region.as_mask(shape=(yshape, xshape),
                                       zero_center=zero_center)
            else:
                region_mask = np.tile(twoD_mask, (nchans, 1, 1))

                # Now blank the channels where the mask isn't there
                region_mask[:start] = \
                    np.zeros((start, yshape, xshape), dtype=bool)
                region_mask[end:] = \
                    np.zeros((nchans - end + 1, yshape, xshape),
                             dtype=bool)

        # Multiply by the mask to remove potential empty regions in the shell.
        # The hole masks are defined where there isn't signal, so multiple by
        # not mask
        if mask is not None:
            region_mask *= mask

        return region_mask

    def as_shell_annulus(self, area_factor=2, zero_center=False):
        '''
        Finds the shell region associated with the bubble.

        Parameters
        ----------
        area_factor : float
            Number of times the area of the bubble where shell regions can be
            considered.
        '''

        if area_factor < 1:
            raise TypeError("The area factor must be >=1.")

        # We're going to assume that each dimension is extend by the same
        # factor. So each axis is extended by sqrt(area_factor)

        shell_annulus = \
            self.as_ellipse(zero_center=zero_center,
                            extend_factor=np.sqrt(area_factor)) - \
            self.as_ellipse(zero_center=zero_center)

        return shell_annulus

    def as_shell_mask(self, mask=None, shape=None, include_center=True,
                      zero_center=False, area_factor=2, spectral_extent=False,
                      use_twoD_regions=False):
        '''
        Realize the shell region as a boolean mask.

        Parameters
        ----------
        include_center : bool, optional
            When enabled, includes mask regions within the hole region. By
            default, this is enabled to ensure small regions are not cut-off.
            This is only applied when a mask is provided (otherwise the notion
            of the shell region doesn't make much sense).

        '''

        # This situation seems ill-defined unless the mask shape is assumed.
        # Force to the mask shape in this case.
        if shape is None and mask is not None:
            shape = mask.shape

        if use_twoD_regions and shape is None and mask is None:
            raise NotImplementedError("Issues with defining a single grid for"
                                      " each local region.")

        # Returns the bbox shape. Forces zero_center to be True
        if shape is None:
            zero_center = True
            bbox = self.as_ellipse(zero_center=False,
                                   extend_factor=np.sqrt(area_factor)).\
                bounding_box

            y_range = ceil_int((bbox[0][1] - bbox[0][0]))
            x_range = ceil_int((bbox[1][1] - bbox[1][0]))

            yy, xx = np.mgrid[-int(y_range / 2): int(y_range / 2) + 1,
                              -int(x_range / 2): int(x_range / 2) + 1]

        else:
            if len(shape) == 2:
                yshape, xshape = shape
            elif len(shape) == 3:
                yshape, xshape = shape[1:]
            else:
                raise TypeError("shape must be for 2D or 3D.")

            yy, xx = np.mgrid[:yshape, :xshape]

        if include_center and mask is not None:
            twoD_mask = \
                self.as_ellipse(zero_center=zero_center,
                                extend_factor=np.sqrt(area_factor))(xx, yy).\
                astype(np.bool)
        else:
            if include_center:
                warn("include_mask is only used when a mask is given.")
            twoD_mask = \
                self.as_shell_annulus(zero_center=zero_center,
                                      area_factor=area_factor)(xx, yy).\
                astype(np.bool)

        # Just return the 2D footprint
        if not spectral_extent:
            shell_mask = twoD_mask
        elif spectral_extent and isinstance(self, Bubble2D):
            raise TypeError("Can only use spectral_extent for Bubble3D"
                            " objects.")
        else:

            if shape is None:
                nchans = self.channel_width
                start = 0
                end = self.channel_width + 1
                yshape = yy.shape[0]
                xshape = yy.shape[1]
            else:
                if len(shape) != 3:
                    raise TypeError("A 3D shape must be given when returning"
                                    " a 3D mask.")
                nchans = shape[0]
                start = self.channel_start
                end = self.channel_end + 1
                yshape = shape[1]
                xshape = shape[2]

            if use_twoD_regions:
                shell_mask = \
                    np.zeros((nchans, yshape, xshape),
                             dtype=np.bool)

                chans = np.arange(start, end).astype(int)
                for i, region in zip(chans, self._twoD_region_iter()):
                    shell_mask[i] = \
                        region.as_shell_mask(shape=(yshape, xshape),
                                             zero_center=zero_center,
                                             area_factor=area_factor)
            else:
                shell_mask = np.tile(twoD_mask, (nchans, 1, 1))

                # Now blank the channels where the mask isn't there
                shell_mask[:start] = \
                    np.zeros((start, yshape, xshape), dtype=bool)
                shell_mask[end + 1:] = \
                    np.zeros((nchans - (end + 1), yshape, xshape),
                             dtype=bool)

        # Multiply by the mask to remove potential empty regions in the shell.
        # The hole masks are defined where there isn't signal, so multiple by
        # not mask
        if mask is not None:
            shell_mask *= ~mask

        return shell_mask


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
    def angular_std(self):
        '''
        The angular standard deviation of the shell positions.
        '''
        return self._angular_std

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

    def __repr__(self):
        s = "2D Bubble at: ({0:6f}, {1:6f})\n".format(self.y, self.x)
        if self.major == self.minor:
            s += "Radius: {0:6f} \n".format(self.major)
        else:
            s += "Major radius: {0:6f} \n".format(self.major)
            s += "Minor radius: {0:6f} \n".format(self.minor)
            s += "Position Angle: {0:6f} \n".format(self.pa)

        if hasattr(self, "_shell_fraction") is not None:
            shell = "Closed" if self.is_closed else "Partial"
            s += "{0} shell with fraction of: {1:6f}" \
                " \n".format(shell, self.shell_fraction)

        return s


class Bubble3D(BubbleNDBase):
    """
    3D Bubbles.

    Parameters
    ----------
    cube : SpectralCube
        Uses the cube to find the spatial and spectral extents of the bubble.
    """
    def __init__(self, props, cube=None, twoD_regions=None, mask=None,
                 distance=None, **kwargs):
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

        if distance is not None:
            self._distance = distance.to(u.pc)

        if twoD_regions is not None:
            # Define the shell fraction as the maximum from the 2D regions.
            # There is some cool stuff to be done with
            fracs = np.array([reg.shell_fraction for reg in self.twoD_regions])
            self._shell_fraction = np.max(fracs)

        # Set the bubble type
        if cube is not None and mask is not None:
            self.find_bubble_type(cube, mask, **kwargs)
            self.set_shell_properties(cube, mask)

        if cube is not None:
            self.set_wcs_extents(cube)

    @staticmethod
    def from_2D_regions(twod_region_list, refit=True,
                        cube=None, mask=None, **fit_kwargs):
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
            # Leave the circles out of the PA calculation
            pas = twoD_properties[:, 4][np.nonzero(twoD_properties[:, 4])]
            if pas.size == 0:
                # All circles. No reason to average
                av_pa = 0.0
            else:
                # circular mean is defined on a 2 pi range. The PAs have a
                # range of pi. So multiply by 2 to average, then divide by
                # half, before finally
                av_pa = wrap_to_pi(0.5 * circmean(2 * pas))
            props = np.array([np.median(twoD_properties[:, 0]),
                              np.median(twoD_properties[:, 1]),
                              np.median(twoD_properties[:, 2]),
                              np.median(twoD_properties[:, 3]),
                              av_pa])

        props = np.append(props,
                          [int(round(np.median(twoD_properties[:, 5]))),
                           int(twoD_properties[:, 5].min()),
                           int(twoD_properties[:, 5].max())])

        self = Bubble3D(props, cube=cube, mask=mask,
                        twoD_regions=twod_region_list)

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
        no_wcs_warning()

    @property
    def velocity_end(self):
        if self._velocity_end is not None:
            return self._velocity_end
        no_wcs_warning()

    @property
    def velocity_center(self):
        if self._velocity_center is not None:
            return self._velocity_center
        no_wcs_warning()

    @property
    def velocity_width(self):
        if self._vel_width is not None:
            return self.channel_width * self._vel_width
        no_wcs_warning()

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
    def shell_velocity_mean(self):
        return self._shell_velocity_mean

    @property
    def shell_velocity_disp(self):
        return self._shell_velocity_disp

    @property
    def expansion_velocity(self):
        return self._expansion_velocity

    def tkin(self, prefactor=0.978, age_unit=u.Myr):
        '''
        Upper limit for the kinetic age of the hole. The prefactor reflects a
        correction for a higher expansion velocity at earlier stages of
        expansion. 0.978 is given in Bagetakos et al. (2011).
        '''

        raise NotImplementedError("Need to support physical units first.")

        tkin = prefactor * 0.5 * self.diameter / self.expansion_velocity
        return tkin.to(age_unit)

    @property
    def bubble_type(self):
        return self._bubble_type

    @bubble_type.setter
    def bubble_type(self, input_type):
        '''
        Must be 1 (blowout), 2 (partial blowout), or 3 (enclosed).
        '''
        if input_type not in [1, 2, 3, 4]:
            raise TypeError("Bubble type must be 1, 2, 3 or 4.")

        self._bubble_type = input_type

    def find_bubble_type(self, cube, mask, min_frac_filled=0.5):
        '''
        Use the cube to determine what type of bubble this is.

        There are 3 types:
         1) blow-out - no significant emission before or after bubble region
                       (spectrally)
         2) half blow-out - one side of the bubble is bounded, the other
                            extends beyond channels with significant emission.
         3) bounded - both sides of the bubble are bounded by regions with
                      significant emission.

        '''

        # We want to check whether the next channel in each direction contain
        # a similar intensity level as found within the hole.

        # mean, std = self.intensity_props(cube)

        # Look before
        if self.channel_start == 0:
            warn("Bubble starts on the first channel. We're going to assume"
                 " that this constitutes a blow-out.")
            before_blowout = True
        else:
            before_channel = self.channel_start - 1

            # Check the mask to see if the majority is not a bubble region.
            twoD_mask = self.as_mask(mask=~mask[before_channel])

            frac_filled = twoD_mask.sum() / np.floor(self.area)

            if frac_filled >= min_frac_filled:
                before_blowout = False
            else:
                before_blowout = True

        # Now look after
        if self.channel_end == cube.shape[0]:
            warn("Bubble ends on the last channel. We're going to assume"
                 " that this constitutes a blow-out.")
            end_blowout = True
        else:
            end_channel = self.channel_end + 1

            # Check the mask to see if the majority is not a bubble region.
            twoD_mask = self.as_mask(mask=~mask[end_channel])

            frac_filled = twoD_mask.sum() / np.floor(self.area)

            if frac_filled >= min_frac_filled:
                end_blowout = False
            else:
                end_blowout = True

        if before_blowout and end_blowout:
            self.bubble_type = 1
        elif before_blowout and not end_blowout:
            self.bubble_type = 2
        elif not before_blowout and end_blowout:
            self.bubble_type = 3
        else:
            self.bubble_type = 4

    def find_expansion_velocity(self, cube):
        '''
        Calculate the expansion velocity, using the definitions from
        Bagetakos et al. (2011). This depends on the bubble type, so that
        must be set. The mean velocity and dispersion within the shell must
        also be set.
        '''

        # These both require more than the cube, so raise a warning instead of
        # trying to run these functions
        if not hasattr(self, "_bubble_type"):
            raise Warning("Run find_bubble_type first.")
        elif not hasattr(self, "__shell_velocity_disp"):
            raise Warning("Run set_shell_properties first.")

        # For complete blowouts, assume that the final expansion velocity is
        # equal to the velocity dispersion in the gas. Use the dispersion
        # within the shell.
        if self.bubble_type == 1:
            self._expansion_velocity = self.shell_velocity_disp
        # Half blowouts use the difference between the mean gas velocity in
        # the shell and the channel velocity of the bounded side.
        elif self.bubble_type == 2:
            self._expansion_velocity = \
                np.abs(self.shell_velocity_mean - self.velocity_end)
        elif self.bubble_type == 3:
            self._expansion_velocity = \
                np.abs(self.shell_velocity_mean - self.velocity_start)
        # And bounded bubbles use half of the difference between their extents
        else:
            self._expansion_velocity =  \
                0.5 * np.abs(self.velocity_start - self.velocity_end)

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

            subcube = self.slice_to_bubble(cube, **kwargs)

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

        ## Return a rectangle (side-view of a cylinder) when the bubble
        # is a blowout.

        # if self.bubble_type == 1:
        #     return Rectangle(...)
        # else:
        return Ellipse((x_cent, chan_cent),
                       width=2 * self.major,
                       height=self.channel_width,
                       angle=0.0, **kwargs)

    # def as_mask(self, spatial_shape, zero_center=False):
    #     '''
    #     Return an elliptical mask.
    #     '''

    #     if len(spatial_shape) != 2:
    #         raise ValueError("spatial_shape must have a length of 2.")

    #     if not self.has_2D_regions:
    #         raise NotImplementedError("")

    #     ellip_mask = np.zeros((len(self.twoD_regions),) + spatial_shape,
    #                           dtype=bool)

    #     for i, region in enumerate(self._twoD_region_iter()):
    #         ellip_mask[i] = \
    #             region.as_mask(shape=spatial_shape,
    #                            zero_center=zero_center)

    #     return ellip_mask

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

    def visualize_shell(self, ax=None, fig=None):
        '''
        Make a 3D point plot of the shell coordinates.
        '''

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as p

        if fig is None:
            fig = p.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.shell_coords[:, 2], self.shell_coords[:, 1],
                   self.shell_coords[:, 0])

        # Flush this out with proper labels, etc...
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("v")

        p.show()

    def visualize(self, cube, return_plot=False, use_aplpy=True,
                  spatial_pad=10, spec_pad=5):
        '''
        Visualize the bubble within the cube.
        '''

        moment0 = self.return_moment0(cube, spatial_pad=spatial_pad,
                                      spec_pad=spec_pad)

        pvslice = \
            self.extract_pv_slice(cube, spatial_pad=spatial_pad,
                                  spec_pad=spec_pad)

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

        if hasattr(self, "_shell_fraction") is not None:
            shell = "Closed" if self.is_closed else "Partial"
            s += "{0} shell with fraction of: {1:6f}" \
                " \n".format(shell, self.shell_fraction)

        return s
