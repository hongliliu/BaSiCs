
import numpy as np


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

        from basics.profile import azimuthal_profiles

        return azimuthal_profiles(array, self.params, **kwargs)

    def find_shell_fraction(self, array, value_thresh=0.0,
                            grad_thresh=1, **kwargs):
        '''
        Find the fraction of the bubble edge associated with a shell.
        '''

        shell_frac = 0
        ntheta = 0

        for prof in self.profile_lines(array, **kwargs):

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

    def as_mask(self):
        '''
        Return a boolean mask of the 2D region.
        '''
        raise NotImplementedError()

    def find_shape(self):
        '''
        Expand/contract to match the contours in the data.
        '''
        raise NotImplementedError()
