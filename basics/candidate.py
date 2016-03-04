
import numpy as np


class Candidate(object):
    """
    Class for candidate bubble portions from 2D planes.
    """
    def __init__(self, mask, img_coords):
        super(Candidate, self).__init__()
        self.mask = mask
        self.img_coords = img_coords

        self._parent = None
        self._child = None

    def get_props(self):
        '''
        Properties of the bubble candidate.
        '''

        self._size = self.mask.sum()

        self._pa = None
        self._major = None
        self._minor = None

    @property
    def size(self):
        return self._size

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


class CandidateInteraction(object):
    """
    Common properties between candidates based on their hierarchal structure
    """
    def __init__(self, candidate1, candidate2):
        super(CandidateInteraction, self).__init__()
        self.candidate1 = candidate1
        self.candidate2 = candidate2

