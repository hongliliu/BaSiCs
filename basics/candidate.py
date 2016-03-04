
import numpy as np


class Bubble2D(object):
    """
    Class for candidate bubble portions from 2D planes.
    """
    def __init__(self, props):
        super(Bubble2D, self).__init__()

        self._channel = props[0]
        self._y = props[1]
        self._x = props[2]
        self._major = props[3]
        self._minor = props[4]
        self._pa = props[5]

    @property
    def params(self):
        return [self._channel, self._y, self._x, self._major,
                self._minor, self._pa]

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

    def profiles_lines(self, array, **kwargs):
        '''
        Calculate radial profile lines of the 2D bubbles.
        '''

        from basics.profile import azimuthal_profiles

        return azimuthal_profiles(array, self.params, **kwargs)

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
