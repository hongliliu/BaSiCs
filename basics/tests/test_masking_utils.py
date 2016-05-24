import pytest
import numpy as np
from astropy.modeling.models import Ellipse2D

from basics.masking_utils import fraction_in_mask


def test_all_in_fraction():

    blob = [50., 50., 20.]
    mask = np.ones((100, 100))

    assert fraction_in_mask(blob, mask) == 1.0


def test_none_in_fraction():

    blob = [50., 50., 20.]
    mask = np.zeros((100, 100))

    assert fraction_in_mask(blob, mask) == 0.0


def test_half_ellipse_in_fraction():

    blob = [50., 50., 20., 10., 0.]

    mask = np.zeros((101, 101))
    mask[:50] = 1

    np.testing.assert_allclose(fraction_in_mask(blob, mask), 0.48,
                               rtol=0.01)


def test_half_ellipse_in_fraction_rotate():

    blob = [70., 30., 20., 10., np.pi / 4.]

    ellip = Ellipse2D(True, blob[1], blob[0], blob[2], blob[3], blob[4])

    yy, xx = np.mgrid[:101, :1011]
    mask = ellip(yy, xx).astype(bool)

    np.testing.assert_allclose(fraction_in_mask(blob, mask), 1.0,
                               rtol=0.01)
