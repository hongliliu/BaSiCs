
import numpy as np
from astropy.modeling.models import Ellipse2D
from astropy.stats import circmean
from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from spectral_cube.lower_dimensional_structures import LowerDimensionalObject
from spectral_cube import SpectralCube
from warnings import warn

from log import overlap_metric
from utils import (floor_int, ceil_int, wrap_to_pi, robust_skewed_std,
                   check_give_beam)
from fan_pvslice import pv_wedge, warp_ellipse_to_circle
from fit_models import fit_region
from galaxy_utils import galactic_radius


def no_wcs_warning():
    warn("No data product with an attached WCS was provided.")


def no_distance_warning():
    warn("No distance was provided.")


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
    def major_physical(self):
        if hasattr(self, "_major_physical"):
            return self._major_physical
        no_distance_warning()

    @property
    def major_angular(self):
        if self._major_angular is not None:
            return self._major_angular
        no_wcs_warning()

    @property
    def minor(self):
        return self._minor

    @property
    def minor_physical(self):
        if hasattr(self, "_minor_physical"):
            return self._minor_physical
        no_distance_warning()

    @property
    def minor_angular(self):
        if self._minor_angular is not None:
            return self._minor_angular
        no_wcs_warning()

    @property
    def diameter(self):
        return 2 * np.sqrt(self.major * self.minor)

    @property
    def diameter_physical(self):
        try:
            return 2 * np.sqrt(self.major_physical * self.minor_physical)
        except TypeError:
            no_distance_warning()

    @property
    def diameter_angular(self):
        try:
            return 2 * np.sqrt(self.major_angular * self.minor_angular)
        except TypeError:
            no_wcs_warning()

    @property
    def eccentricity(self):
        return self.major / float(self.minor)

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):

        if not hasattr(value, "unit"):
            raise TypeError("distance must be given as an astropy Quantitiy"
                            " with an appropriate unit.")

        if not value.unit.is_equivalent(u.pc):
            raise ValueError("distance must be given with an appropriate unit"
                             " of distance.")

        self._distance = value.to(u.kpc)

    def galactocentric_radius(self, galaxy_coord, pa, inc, unit=u.kpc):
        '''
        Requires a few galaxy properties
        '''

        if not hasattr(self, "_distance"):
            raise ValueError("distance must be provided to find the "
                             "galactocentric radius.")

        return galactic_radius(self.center_coordinate, galaxy_coord,
                               self.distance, pa, inc)

    @property
    def center_coordinate(self):
        return SkyCoord(self.ra, self.dec, unit=(u.deg, u.deg))

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

    def intensity_props(self, data, area_factor=1., region='hole', mask=None,
                        robust_estimate=True):
        '''
        Return the mean and std for the elliptical region in the given data.

        Parameters
        ----------
        robust_estimate : bool, optional
            Estimate standard deviation using the 2.5th and 15th percentiles.
            This is ideal for a near-normal distribution skewed to high
            values.

        '''

        if isinstance(data, LowerDimensionalObject):
            is_2D = True
        elif isinstance(data, SpectralCube):
            is_2D = False
        else:
            raise TypeError("data must be a LowerDimensionalObject or "
                            "SpectralCube.")

        if mask is not None:
            if not is_2D and len(mask.shape) != 3:
                raise TypeError("Must provide a 3D mask when providing 3D "
                                "data.")

        if region == "hole":
            local_mask, slices = self.as_mask(mask=mask, minimal_shape=True)
        elif region == "shell":
            local_mask, slices = self.as_shell_mask(mask=mask,
                                                    minimal_shape=True)
        else:
            raise TypeError("region must be 'hole' or 'shell'")

        if is_2D:
            if robust_estimate:
                return robust_skewed_std(data[slices].value[local_mask])

            return np.nanmean(data[slices].value[local_mask]), \
                np.nanstd(data[slices].value[local_mask])
        else:
            if robust_estimate:
                return robust_skewed_std(data.filled_data[slices].value[local_mask])

            return np.nanmean(data.filled_data[slices][local_mask]), \
                np.nanstd(data.filled_data[slices][local_mask])

    def find_hole_contrast(self, data, mask=None, area_factor=2.,
                           noise_std=None):
        '''
        Relative difference between the mean intensity in the hole versus the
        shell. Sort of similar to Column 8 in Table 1 of Deul & den Hartog
        (1990).
        '''

        shell_mean, shell_std = \
            self.intensity_props(data, area_factor=area_factor,
                                 region='shell', mask=mask,
                                 robust_estimate=True)

        hole_mean, hole_std = \
            self.intensity_props(data, area_factor=1.,
                                 region='hole', mask=mask,
                                 robust_estimate=True)

        # Since a lot of the holes will have a bkg at/near the noise limit,
        # use the std when the hole mean is less than hole std. Low values
        # may result from a lack of samples of the noise, but we should
        # assume the complete noise distribution is normal with mean 0.
        # The true noise std can be provided to avoid this.
        if noise_std is not None:
            if hole_mean < noise_std:
                self._hole_contrast = \
                    (shell_mean - noise_std) / shell_mean
        elif hole_mean < 0.0:
            self._hole_contrast = ((shell_mean - hole_std) / shell_mean)
        else:
            self._hole_contrast = ((shell_mean - hole_mean) / shell_mean)

    @property
    def hole_contrast(self):
        return self._hole_contrast

    def set_wcs_props(self, data):
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

            # Get the spatial pixel scales. Should be either of the first 2
            # Also must be positive values, so no need for abs
            spat_pix_scale = proj_plane_pixel_scales(data.wcs)[0] * u.deg

            self._major_angular = self.major * spat_pix_scale
            self._minor_angular = self.minor * spat_pix_scale

            # Now set the physical distances, if there distance has been given
            if hasattr(self, "_distance"):
                phys_pix_scale = spat_pix_scale.value * (np.pi / 180.) * \
                    self.distance.to(u.pc)
                self._major_physical = self.major * phys_pix_scale
                self._minor_physical = self.minor * phys_pix_scale

        elif isinstance(data, LowerDimensionalObject):
            # At some point, the 2D LDO will also have a
            # spatial_coordinate_map attribute
            raise NotImplementedError("")
        else:
            raise TypeError("data must be a SpectralCube or"
                            " LowerDimensionalObject.")

    def set_shell_properties(self, data, mask, flux_unit=u.K * u.km / u.s,
                             rest_freq=1.141 * u.GHz, **shell_kwargs):
        '''
        Get the properties of the shell
        '''

        # Requires masking for 2D objects in SpectralCube to be ready
        if isinstance(self, Bubble2D):
            raise NotImplementedError("")

        spectral_extent = False
        if isinstance(self, Bubble3D):
            spectral_extent = True

        minimal_shape = True

        if "spectral_extent" in shell_kwargs:
            warn("spectral_extent is automatically set. Ignoring input.")
            del shell_kwargs["spectral_extent"]

        if "minimal_shape" in shell_kwargs:
            warn("minimal_shape is automatically set. Ignoring input.")
            del shell_kwargs["minimal_shape"]

        shell_mask, slices = \
            self.as_shell_mask(mask=mask,
                               spectral_extent=spectral_extent,
                               minimal_shape=minimal_shape,
                               **shell_kwargs)

        # Apply the mask, then cut to its extents
        shell_cube = data[slices].with_mask(shell_mask).minimal_subcube()
        # Once 2D is supported, need extra if/else for minimal_sub-something

        # Can't just use mom0.mean() right now until this issue is dealt with:
        # https://github.com/radio-astro-tools/spectral-cube/issues/279
        # Actually it's even worse than that: nanmean won't work with
        # Projections... Lazy work around it to convert to a Quantity
        mom0 = u.Quantity(shell_cube.moment0())
        self._avg_shell_flux_density = np.nanmean(mom0)
        self._total_shell_flux_density = np.nansum(mom0)

        if not flux_unit.is_equivalent(mom0.unit):
            # I'm going to assume that this case should only arise when
            # converting from Jy/beam to K.

            # Either conversion needs a defined beam
            beam = check_give_beam(data)

            if beam is None:
                raise ValueError("data does not have an attached beam. A beam"
                                 " is needed to convert to"
                                 " {}".format(flux_unit.to_string()))

            jtok = beam.jtok(rest_freq)

            # Only supporting units equivalent to K km/s
            if flux_unit.is_equivalent(u.K * u.m / u.s):
                self._avg_shell_flux_density = \
                    (self._avg_shell_flux_density * jtok / u.Jy).to(flux_unit)
                self._total_shell_flux_density = \
                    (self._total_shell_flux_density * jtok /
                     u.Jy).to(flux_unit)
            else:
                raise TypeError("Only support conversion to units equivalent"
                                " to K km/s.")
        else:
            self._avg_shell_flux_density = \
                self._avg_shell_flux_density.to(flux_unit)
            self._total_shell_flux_density = \
                self._total_shell_flux_density.to(flux_unit)

        # In 3D, set the velocity properties
        if isinstance(self, Bubble3D):
            self._shell_velocity_mean = \
                np.nanmean(u.Quantity(shell_cube.moment1()))
            # Define the dispersion as half the FWHM linewidth
            self._shell_velocity_disp = \
                np.nanmean(u.Quantity(0.5 * shell_cube.linewidth_fwhm()))

    @property
    def avg_shell_flux_density(self):
        return self._avg_shell_flux_density

    @property
    def total_shell_flux_density(self):
        return self._total_shell_flux_density

    @property
    def shell_column_density(self):
        return 1.823e18 * self.avg_shell_flux_density.value / u.cm**2

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
                spectral_extent=False, use_twoD_regions=False,
                minimal_shape=False):
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

        model_ellipse = self.as_ellipse(zero_center=False)

        if minimal_shape or shape is None:
            bbox = model_ellipse.bounding_box
            yextents = (floor_int(bbox[0][0]), ceil_int(bbox[0][1]) + 1)
            xextents = (floor_int(bbox[1][0]), ceil_int(bbox[1][1]) + 1)
        else:
            if len(shape) == 2:
                yshape, xshape = shape
            elif len(shape) == 3:
                yshape, xshape = shape[1:]
            else:
                raise TypeError("shape must be for 2D or 3D.")
            yextents = (0, yshape)
            xextents = (0, xshape)

        # Returns the bbox shape. Forces zero_center to be True
        if shape is None or minimal_shape:
            yy, xx = np.mgrid[yextents[0]: yextents[1],
                              xextents[0]: xextents[1]]
        else:
            yy, xx = np.mgrid[:yshape, :xshape]

        yshape = yy.shape[0]
        xshape = yy.shape[1]

        twoD_mask = model_ellipse(xx, yy).astype(np.bool)

        # Just return the 2D footprint
        if mask is not None:
            spectral_extent = spectral_extent if len(mask.shape) == 2 else True

        if shape is None:
            slices = (slice(0, yshape),
                      slice(0, xshape))
        else:
            slices = (slice(yextents[0], yextents[1]),
                      slice(xextents[0], xextents[1]))

        if not spectral_extent:
            region_mask = twoD_mask
        elif spectral_extent and isinstance(self, Bubble2D):
            raise TypeError("Can only use spectral_extent for Bubble3D"
                            " objects.")
        else:

            if minimal_shape or shape is None:
                nchans = self.channel_width
                start = 0
                end = self.channel_width
            else:
                if len(shape) != 3:
                    raise TypeError("A 3D shape must be given when returning"
                                    " a 3D mask.")
                nchans = shape[0]
                start = self.channel_start
                end = self.channel_end + 1

            if use_twoD_regions:
                region_mask = \
                    np.zeros((nchans, yshape, xshape),
                             dtype=np.bool)

                chans = np.arange(start, end).astype(int)
                for i, region in zip(chans, self._twoD_region_iter()):
                    region_mask[i] = \
                        region.as_mask(shape=(yshape, xshape),
                                       zero_center=zero_center,
                                       minimal_shape=False)
            else:
                region_mask = np.tile(twoD_mask, (nchans, 1, 1))

                # Now blank the channels where the mask isn't there
                region_mask[:start] = \
                    np.zeros((start, yshape, xshape), dtype=bool)
                region_mask[end:] = \
                    np.zeros((nchans - end, yshape, xshape),
                             dtype=bool)

            if minimal_shape:
                slices = (slice(self.channel_start, self.channel_end + 1), ) \
                    + slices
            else:
                slices = (slice(0, nchans), ) + slices

        # Multiply by the mask to remove potential empty regions in the shell.
        # The hole masks are defined where there isn't signal, so multiple by
        # not mask
        if mask is not None:
            region_mask *= mask[slices]

        if minimal_shape:
            return region_mask, slices
        else:
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
                      use_twoD_regions=False, minimal_shape=False):
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

        if include_center and mask is not None:
            model_shell = \
                self.as_ellipse(zero_center=zero_center,
                                extend_factor=np.sqrt(area_factor))
        else:
            if include_center:
                warn("include_mask is only used when a mask is given.")
            model_shell = \
                self.as_shell_annulus(zero_center=zero_center,
                                      area_factor=area_factor)

        if minimal_shape or shape is None:
            bbox = self.as_ellipse(zero_center=False,
                                   extend_factor=np.sqrt(area_factor)).\
                bounding_box
            yextents = (floor_int(bbox[0][0]), ceil_int(bbox[0][1]) + 1)
            xextents = (floor_int(bbox[1][0]), ceil_int(bbox[1][1]) + 1)
        else:
            if len(shape) == 2:
                yshape, xshape = shape
            elif len(shape) == 3:
                yshape, xshape = shape[1:]
            else:
                raise TypeError("shape must be for 2D or 3D.")
            yextents = (0, yshape)
            xextents = (0, xshape)

        # Returns the bbox shape. Forces zero_center to be True
        if shape is None or minimal_shape:
            yy, xx = np.mgrid[yextents[0]: yextents[1],
                              xextents[0]: xextents[1]]
        else:
            yy, xx = np.mgrid[:yshape, :xshape]

        yshape = yy.shape[0]
        xshape = yy.shape[1]

        twoD_mask = model_shell(xx, yy).astype(np.bool)

        # Just return the 2D footprint
        if mask is not None:
            spectral_extent = spectral_extent if len(mask.shape) == 2 else True

        if shape is None:
            slices = (slice(0, yshape),
                      slice(0, xshape))
        else:
            slices = (slice(yextents[0], yextents[1]),
                      slice(xextents[0], xextents[1]))

        if not spectral_extent:
            shell_mask = twoD_mask
        elif spectral_extent and isinstance(self, Bubble2D):
            raise TypeError("Can only use spectral_extent for Bubble3D"
                            " objects.")
        else:

            if minimal_shape or shape is None:
                nchans = self.channel_width
                start = 0
                end = self.channel_width
            else:
                if len(shape) != 3:
                    raise TypeError("A 3D shape must be given when returning"
                                    " a 3D mask.")
                nchans = shape[0]
                start = self.channel_start
                end = self.channel_end

            if use_twoD_regions:
                shell_mask = \
                    np.zeros((nchans, yshape, xshape),
                             dtype=np.bool)

                chans = np.arange(start, end).astype(int)
                for i, region in zip(chans, self._twoD_region_iter()):
                    shell_mask[i] = \
                        region.as_mask(shape=(yshape, xshape),
                                       zero_center=zero_center,
                                       minimal_shape=False)
            else:
                shell_mask = np.tile(twoD_mask, (nchans, 1, 1))

                # Now blank the channels where the mask isn't there
                shell_mask[:start] = \
                    np.zeros((start, yshape, xshape), dtype=bool)
                shell_mask[end:] = \
                    np.zeros((nchans - end, yshape, xshape),
                             dtype=bool)

            if minimal_shape:
                slices = (slice(self.channel_start, self.channel_end + 1), ) \
                    + slices
            else:
                slices = (slice(0, nchans), ) + slices

        # Multiply by the mask to remove potential empty regions in the shell.
        # The hole masks are defined where there isn't signal, so multiple by
        # not mask
        if mask is not None:
            shell_mask *= ~mask[slices]

        if minimal_shape:
            return shell_mask, slices
        else:
            return shell_mask


class Bubble2D(BubbleNDBase):
    """
    Class for candidate bubble portions from 2D planes.
    """
    def __init__(self, props, shell_coords=None, channel=None, data=None,
                 distance=None):
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
            self._model_residual = props[8]

        self._shell_coords = shell_coords

        if distance is not None:
            self.distance = distance

        # Requires finishing the WCS extents portion in set_wcs_props
        # if data is not None:
        #     self.set_wcs_props(data)

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
                 distance=None, sigma=None, **kwargs):
        super(Bubble3D, self).__init__()

        self._y = props[0]
        self._x = props[1]
        self._major = props[2]
        self._minor = props[3]
        self._pa = props[4]
        self._channel_center = int(props[5])
        self._channel_start = int(props[6])
        self._channel_end = int(props[7])

        self.twoD_regions = twoD_regions

        if distance is not None:
            self.distance = distance

        if twoD_regions is not None:
            # Define the shell fraction as the maximum from the 2D regions.
            # There is some cool stuff to be done with
            fracs = np.array([reg.shell_fraction for reg in self.twoD_regions])
            self._shell_fraction = np.max(fracs)

        if cube is not None:
            self.set_wcs_props(cube)

        # Set the bubble type
        if cube is not None and mask is not None:
            self.find_bubble_type(cube, mask, **kwargs)
            self.set_shell_properties(cube, mask)
            self.find_expansion_velocity()
            self.find_hole_contrast(cube, mask=mask, noise_std=sigma)

    @staticmethod
    def from_2D_regions(twod_region_list, refit=True,
                        cube=None, mask=None, distance=None, sigma=None,
                        **fit_kwargs):
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
                        twoD_regions=twod_region_list,
                        distance=distance, sigma=sigma)

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

        tkin = prefactor * 0.5 * self.diameter_physical.to(u.km) / \
            self.expansion_velocity.to(u.km / u.s)
        return tkin.to(age_unit)

    def shell_volume_density(self, scale_height=100 * u.pc, inc=55):
        '''
        Bagetakos+11 eqs. 8 & 9

        Only defined in 3D for now. *Technically* getting this from an
        integrated intensity map is fine too, so long as the slice thickness
        is appropriate.
        '''

        # Effective thckness of the HI layer (1-sigma) of the scale height
        # assuming a Gaussian profile
        l_thick = np.sqrt(8 * np.log(2)) * scale_height.to(u.cm) / \
            np.cos(inc)

        return self.shell_column_density / l_thick

    def volume(self, scale_height=None):
        '''
        Definitions from Bagetakos+11 eqs. 10/11. Note that there is an extra
        factor of 2 in Eq. 11 for the cylinder volume.

        For complete blowouts, you need the scale height.
        '''

        if self.bubble_type == 1:
            if scale_height is None:
                raise ValueError("Blowouts require the scale height to"
                                 " compute the volume.")
            l_thick = np.sqrt(8 * np.log(2)) * scale_height.to(u.pc)

            return np.pi * (0.5 * self.diameter_physical) ** 2 * l_thick
        else:
            return (4 * np.pi / 3.) * (0.5 * self.diameter_physical) ** 3

    def hole_mass(self, scale_height=100. * u.pc, inc=55):
        '''
        Calculate the approximate mass of HI evacuated from the hole. This
        relies on the volume and the midplane volume density.

        Bagetakos+11 eq. 12
        '''

        # I can't reproduce the Bagetakos mass values using eq. 12 using their
        # values of the diameter (for volume) and nHI. Since nHI * V gives the
        # number of hydrogen atoms, just convert straight to the mass
        mass_factor = (1.67e-27 * u.kg).to(u.Msun)

        return mass_factor * self.shell_volume_density(scale_height, inc) * \
            self.volume(scale_height).to(u.cm**3)

    def formation_energy(self, scale_height=100. * u.pc, inc=55):
        '''
        Chevalier's equation for the energy needed to drive an expanding
        shell. Using the form as shown in Bagetakos+11 eq. 18.

        '''

        vol_dens = np.power(self.shell_volume_density(scale_height, inc).value,
                            1.12)
        size = np.power(0.5 * self.diameter_physical.value, 3.12)
        exp_vel = np.power(self.expansion_velocity.to(u.km / u.s).value, 1.4)

        return 5.3e43 * vol_dens * size * exp_vel * u.erg

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

    def find_bubble_type(self, cube, mask, min_frac_filled=0.5, nsig=2):
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

        hole_mean, hole_sig = \
            self.intensity_props(cube, mask=mask, region='hole')

        # Look before
        if self.channel_start == 0:
            warn("Bubble starts on the first channel. We're going to assume"
                 " that this constitutes a blow-out.")
            before_blowout = True
        else:
            before_channel = self.channel_start - 1

            # Check how much of the previous channel within the bubble mask
            # is above the hole intensity threshold
            twoD_mask, slices = \
                self.as_mask(shape=cube.shape[1:], minimal_shape=True)

            # Add the velocity channel
            slices = (before_channel, ) + slices

            region_mask = cube[slices].value > \
                nsig * hole_sig + hole_mean

            frac_filled = (twoD_mask * region_mask).sum() / np.floor(self.area)

            if frac_filled >= min_frac_filled:
                before_blowout = False
            else:
                before_blowout = True

        # Now look after
        if self.channel_end == cube.shape[0] - 1:
            warn("Bubble ends on the last channel. We're going to assume"
                 " that this constitutes a blow-out.")
            end_blowout = True
        else:
            end_channel = self.channel_end + 1

            # Check how much of the previous channel within the bubble mask
            # is above the hole intensity threshold
            twoD_mask, slices = \
                self.as_mask(shape=cube.shape[1:], minimal_shape=True)

            # Add the velocity channel
            slices = (end_channel, ) + slices

            region_mask = cube[slices].value > \
                nsig * hole_sig + hole_mean

            frac_filled = (twoD_mask * region_mask).sum() / np.floor(self.area)

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

    def find_expansion_velocity(self):
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
        elif not hasattr(self, "_shell_velocity_disp"):
            raise Warning("Run set_shell_properties first.")

        # For complete blowouts, assume that the final expansion velocity is
        # equal to the velocity dispersion in the gas. Use the dispersion
        # within the shell.
        if self.bubble_type == 1:
            self._expansion_velocity = self.shell_velocity_disp
        # Half blowouts use the difference between the mean gas velocity in
        # the shell and the channel velocity of the bounded side.
        # NOTE: the Bagetakos Vexp definition for half blow-outs is giving
        # very small values (>> channel width). We're going to adopt the same
        # form as the enclosed shells. Given that the uncertainty is capped
        # at the velocity resolution, this assumption does not seem significant

        # elif self.bubble_type == 2:
        #     self._expansion_velocity = \
        #         np.abs(self.shell_velocity_mean - self.velocity_end)
        # elif self.bubble_type == 3:
        #     self._expansion_velocity = \
        #         np.abs(self.shell_velocity_mean - self.velocity_start)
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

    def extract_pv_slice(self, cube, width=None, use_subcube=True,
                         warp_to_circle=True, **kwargs):
        '''
        Return a PV Slice. Defaults to across the entire bubble.
        '''

        try:
            import pvextractor
        except ImportError:
            raise ImportError("pvextractor must be installed to extract "
                              " PV slices.")

        if warp_to_circle:
            if not use_subcube:
                raise TypeError("Due to heavy memory usage, warping can"
                                " only be used with 'use_subcube=True'.")
            if self.eccentricity < 1.2:
                warn("Bubble is nearly circular. Skipping the warp.")
                warp_to_circle = False

        if "spatial_pad" in kwargs:
            spatial_pad = kwargs["spatial_pad"]
        else:
            spatial_pad = 0

        if width is None:
            width = 1  # self.minor

        if use_subcube:

            subcube = self.slice_to_bubble(cube, **kwargs)

            if warp_to_circle:
                subcube = warp_ellipse_to_circle(subcube, self.major,
                                                 self.minor, self.pa)
                max_dist = 2 * float(self.minor + spatial_pad)
            else:
                # Use major here. Will be find ~ circular regions.
                max_dist = 2 * float(self.major + spatial_pad)

            sub_center = (floor_int(subcube.shape[1] / 2.),
                          floor_int(subcube.shape[2] / 2.))

            return pv_wedge(subcube, sub_center, max_dist, 0.0, np.pi,
                            width=width)
        else:
            # Define end points along the major axis
            max_dist = 2 * float(self.major + spatial_pad)

            return pv_wedge(cube, self.center, max_dist, 0.0, np.pi,
                            width=width)

    def as_pv_patch(self, x_cent=None, chan_cent=None, **kwargs):
        '''
        Return a PV slice. Aligns the direction along the major axis.
        '''
        from matplotlib.patches import Ellipse, Rectangle

        if x_cent is None:
            x_cent = self.major

        if chan_cent is None:
            chan_cent = self.channel_center

        # Return a rectangle (side-view of a cylinder) when the bubble
        # is a blowout.
        if self.bubble_type == 1:
            return Rectangle((x_cent, chan_cent), 2 * self.major,
                             self.channel_width, angle=0.0, **kwargs)
        else:
            return Ellipse((x_cent, chan_cent),
                           width=2 * self.major,
                           height=self.channel_width,
                           angle=0.0, **kwargs)

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
        s = "Type {0} Bubble at: ({1:6f}, {2:6f}," \
            "{3:6f})\n".format(self.bubble_type, self.channel_center,
                               self.y, self.x)
        if self.major == self.minor:
            s += "Radius: {0:6f} \n".format(self.major)
        else:
            s += "Major radius: {0:6f} \n".format(self.major)
            s += "Minor radius: {0:6f} \n".format(self.minor)
            s += "Position Angle: {0:6f} \n".format(self.pa)

        s += "Channel width: {0:6f} \n".format(self.channel_width)
        s += "Spectral width: {0:6f} {1} \n".format(self.velocity_width.value,
                                                    self.velocity_width.unit.to_string())

        if hasattr(self, "_shell_fraction") is not None:
            shell = "Closed" if self.is_closed else "Partial"
            s += "{0} shell with fraction of: {1:6f}" \
                " \n".format(shell, self.shell_fraction)

        return s
