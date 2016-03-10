
import numpy as np
from astropy.modeling.models import Ellipse2D
from astropy.nddata.utils import extract_array, add_array
from scipy import ndimage as nd

from log import overlap_metric
from utils import consec_split, find_nearest
from profile import _line_profile_coordinates

eight_conn = np.ones((3, 3))


class Bubble2D(object):
    """
    Class for candidate bubble portions from 2D planes.
    """
    def __init__(self, props):
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

    @property
    def params(self):
        if self.channel:
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
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def center_pixel(self):
        return (self.y, self.x)

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
                   **kwargs):
        '''
        Expand/contract to match the contours in the data.
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
            return add_array(np.zeros_like(array, dtype=bool), extent_mask,
                             self.center_pixel)
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


class Bubble3D(object):
    """
    3D Bubbles.
    """
    def __init__(self, props):
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

    @staticmethod
    def from_2D_regions(twod_region_list):
        '''
        Create a 3D regions from a collection of 2D regions.
        '''

        # Extract the 2D properties
        twoD_properties = \
            np.array([bub2D.params for bub2D in twod_region_list])

        # Sort by channel
        twod_region_list = \
            [twod_region_list[i] for i in twoD_properties[:, 0].argsort()]

        props = [twoD_properties[:, 1].mean(), twoD_properties[:, 2].mean(),
                 twoD_properties[:, 3].max(), twoD_properties[:, 4].max(),
                 twoD_properties[:, 5].mean(),
                 int(round(twoD_properties[:, 0].median())),
                 twoD_properties[:, 0].min(), twoD_properties[:, 0].max()]

        self = Bubble3D(props)

        self._twoD_objects = twod_region_list

        return self

    @property
    def twoD_objects(self):
        return self._twoD_objects

    @property
    def has_2D_regions(self):
        return True if self.twoD_objects is not None else False

    @property
    def y(self):
        return self._y

    @property
    def x(self):
        return self._x

    @property
    def pixel_center(self):
        return (np.floor(self.y).astype(int), np.floor(self.x).astype(int))

    @property
    def major(self):
        return self._major

    @property
    def minor(self):
        return self._minor

    @property
    def pa(self):
        return self._pa

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
        return self.velocity_end - self.velocity_start

    @property
    def bubble_type(self):
        return self._bubble_type

    def extract_pv_slice(self, cube):
        pass

    def as_mask(self, cube):
        pass

    def as_extent_mask(self, cube):
        pass

    def as_shell_mask(self, cube):
        pass

    def return_cube_region(self, cube):
        pass
