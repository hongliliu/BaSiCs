
import numpy as np
import astropy.units as u

from bubble_segment import BubbleSegment


class BubbleFinder(object):
    """docstring for BubbleFinder"""
    def __init__(self, arg):
        super(BubbleFinder, self).__init__()
        self.arg = arg


class BubbleFinder2D(object):
    """
    Get bubbles in a 2D image.
    """
    def __init__(self, array, threshold=None, mask=None, nan_to_zero=True):

        self.array = array
        if nan_to_zero:
            self.array[np.isnan(self.array)] = 0.0

        self.threshold = threshold

        if mask is None:
            self.create_signal_mask(threshold=threshold)
        else:
            self.mask = mask

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask_array):
        if mask_array is None:
            self._mask = np.ones_like(mask_array).astype(bool)
        else:
            if mask_array.shape != self.array.shape:
                raise TypeError("mask must match the shape of the given "
                                "array.")
            self._mask = mask_array

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, input_array):
        if input_array.ndim != 2:
            raise TypeError("Given array must be 2D.")

        self._array = input_array

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):

        if not isinstance(value, u.Quantity):
            raise TypeError("Threshold must be an astropy Quantity.")

        if value.unit not in self.array.unit.find_equivalent_units():
            raise u.UnitsError("Threshold must have equivalent units"
                               " as the array " + str(self.array.unit))

        self._threshold = value

    def create_signal_mask(self, threshold=None):
        '''
        Create a mask given a threshold brightness.
        '''

        if threshold is not None:
            self.threshold = threshold

        if self.threshold is None:
            raise ValueError("Must provide a threshold to create mask.")

        self._mask = self.array > self.threshold

    def create_bubble_mask(self, **kwargs):

        bubbles = BubbleSegment(self.array, mask=self.mask, **kwargs)

        # bubbles.apply_atan_transform()
        bubbles.cut_to_bounding_box()
        bubbles.multiscale_bubblefind()

        self.bubble_mask = bubbles.insert_in_shape(self.array.shape)
