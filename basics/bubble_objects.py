
import numpy as np
from astropy.modeling.models import Ellipse2D

from log import overlap_metric
from utils import consec_split, nearest_posn, find_nearest
from profile import _line_profile_coordinates


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
            return Ellipse2D(True, 0.0, 0.0, 2*self.major, 2*self.minor,
                             self.pa)
        return Ellipse2D(True, self.y, self.x, 2*self.major, 2*self.minor,
                         self.pa)

    def as_mask(self, shape, zero_center=False):
        '''
        Return a boolean mask of the 2D region.
        '''
        yy, xx = np.mgrid[:shape[0], :shape[1]]

        return self.as_ellipse(zero_center=zero_center)(xx, yy)

    def find_shape(self, array, max_extent=1.5, value_thresh=0.0,
                   **kwargs):
        '''
        Expand/contract to match the contours in the data.
        '''

        # Use the ellipse model to define a bounding box for the mask.
        bbox = self.as_ellipse(zero_center=True).bounding_box

        y_range = np.ceil((max_extent * np.sqrt(2)) *
                          (bbox[0][1] - bbox[0][0])).astype(int)
        x_range = np.ceil((max_extent * np.sqrt(2)) *
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

            line_posns = \
                [np.floor(coord).astype(int) + cent for coord, cent in
                 zip(_line_profile_coordinates((0, 0), new_end), centre)]

            # This angle does not coincide with a shell.
            # We fill in a pixel at the major radius.
            if above_thresh.size == 0:
                nearest_idx = find_nearest(dist_arr[line_posns],
                                           self.major).astype(int)
            else:
                # Pick the end point on the first segment with > 2 pixels
                # above the threshold.
                segments = consec_split(above_thresh)

                for seg in segments:
                    if seg.size < 3:
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
                                     self.major).astype(int)

            end_posn = tuple([posn[nearest_idx] for posn in line_posns])

            extent_mask[end_posn] = True

        return extent_mask

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
