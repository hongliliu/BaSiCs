
from basics.log import _ellipse_overlap, _circle_overlap, _min_merge_overlap

import numpy.testing as npt
import numpy as np


def test_circle_overlap():

    pt1 = (10, 10, 10, 10, 0)
    pt2 = (10, 15, 10, 10, 0)

    assert _circle_overlap(pt1, pt2) == 0.68503764247429266
    assert _circle_overlap(pt2, pt1) == 0.68503764247429266


def test_ellipse_overlap():

    pt1 = (10, 10, 10, 10, 0)
    pt2 = (10, 15, 10, 10, 0)

    npt.assert_almost_equal(_circle_overlap(pt1, pt2),
                            _ellipse_overlap(pt1, pt2), decimal=2)


def test_pixel_nooverlap():

    pt1 = (10, 10, 10, 10, 0)
    pt2 = (10, 31, 10, 10, 0)

    npt.assert_almost_equal(0.0,
                            _ellipse_overlap(pt1, pt2), decimal=3)


def test_pixel_alloverlap():

    pt1 = (10, 10, 10, 10, 0)
    pt2 = (10, 15, 20, 20, 0)

    npt.assert_almost_equal(1.0,
                            _ellipse_overlap(pt1, pt2), decimal=3)


def test_min_merge_overlap():

    F = 2.0

    assert _min_merge_overlap(F) == 0.0

    F = 1.0

    npt.assert_almost_equal(_min_merge_overlap(F),
                            (2/3.) - np.sqrt(3)/(2*np.pi))
