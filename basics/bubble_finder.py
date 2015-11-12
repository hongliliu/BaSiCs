

class BubbleFinder(object):
    """docstring for BubbleFinder"""
    def __init__(self, arg):
        super(BubbleFinder, self).__init__()
        self.arg = arg


class BubbleFinder2D(object):
    """
    Get bubbles in a 2D image.
    """
    def __init__(self, array, atan_transform=True, threshold=None, mask=None,
                 cut_to_box=False):
        self.array = array
        self.threshold = threshold
        self.mask = mask

        self._atan_array = None

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

    def apply_atan_transform(self, threshold=None):

        if threshold is not None:
            self._threshold = threshold

        self._atan_array = arctan_transform(self.array, self.threshold)

    @property
    def atan_array(self):
        return self._atan_array

    def get_bounding_box(self):
        pass
