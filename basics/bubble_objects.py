
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

    @property
    def params(self):
        return np.array([self._y, self._x, self._major,
                         self._minor, self._pa])

    @property
    def area(self):
        return np.pi * self.major * self.minor

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
        ntheta = 0

        for dist, prof in self.profile_lines(array, **kwargs):

            # Count number of profiles returned.
            ntheta += 1

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

    def intensity_props(self, array, mask_operation="erode",
                        niters=1):
        '''
        Return the mean and std for the elliptical region in the given array.
        '''

        ellip_mask = self.as_mask(array.shape, zero_center=True)

        if mask_operation is not None:
            if mask_operation is 'erode':
                ellip_mask = nd.binary_erosion(ellip_mask, eight_conn,
                                               iterations=niters)
            elif mask_operation is 'dilate':
                ellip_mask = nd.binary_dilation(ellip_mask, eight_conn,
                                                iterations=niters)

        masked_array = array.copy()
        masked_array[ellip_mask] = np.NaN

        return np.nanmean(masked_array), np.nanstd(masked_array)

    def find_shape(self, array, return_array='full', max_extent=1.5,
                   nsig_thresh=1, value_thresh=None, **kwargs):
        '''
        Expand/contract to match the contours in the data.
        '''

        # Define a suitable background based on the intensity within the
        # elliptical region
        if value_thresh is None:
            mean, std = self.intensity_props(array)
            value_thresh = mean + nsig_thresh * std

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
                                                  append_end=True, **kwargs):

            above_thresh = np.where(prof >= value_thresh)[0]

            new_end = (max_extent * (end[0] - self.y),
                       max_extent * (end[1] - self.x))

            max_dist = np.abs(dist.max())

            line_posns = \
                [np.floor(coord).astype(int) + cent for coord, cent in
                 zip(_line_profile_coordinates((0, 0), new_end), centre)]

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
                    if seg.size < 2:
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
        self._vel_width = props[5]

    @staticmethod
    def from_2D_regions(twod_region_list):
        '''
        Create a 3D regions from a collection of 2D regions.
        '''
        pass
