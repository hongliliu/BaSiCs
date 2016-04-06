
from basics.log import _ellipse_overlap, _circle_overlap, _min_merge_overlap, \
    shell_similarity

import numpy.testing as npt
import numpy as np
import pytest


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

    npt.assert_almost_equal(_ellipse_overlap(pt1, pt2),
                            0.0, decimal=3)


def test_pixel_alloverlap():

    pt1 = (10, 10, 10, 10, 0)
    pt2 = (10, 15, 20, 20, 0)

    npt.assert_almost_equal(_ellipse_overlap(pt1, pt2),
                            1.0, decimal=3)


def test_circle_corr():

    pt1 = (10, 10, 10, 10, 0)
    pt2 = (10, 15, 10, 10, 0)

    Aover = 0.68503764247429266 * np.pi * 10**2
    corr = Aover / (np.pi * 10 * 10)

    npt.assert_almost_equal(_circle_overlap(pt1, pt2, return_corr=True), corr)


def test_ellipse_corr():

    pt1 = (10, 10, 10, 10, 0)
    pt2 = (10, 15, 10, 10, 0)

    npt.assert_almost_equal(_circle_overlap(pt1, pt2, return_corr=True),
                            _ellipse_overlap(pt1, pt2, return_corr=True),
                            decimal=2)


def test_pixel_nooverlap_corr():

    pt1 = (10, 10, 10, 10, 0)
    pt2 = (10, 31, 10, 10, 0)

    npt.assert_almost_equal(_ellipse_overlap(pt1, pt2, return_corr=True),
                            0.0, decimal=3)


def test_pixel_alloverlap_corr():

    pt1 = (10, 10, 10, 10, 0)
    pt2 = (10, 15, 20, 20, 0)

    Aover = _ellipse_overlap(pt1, pt2) * np.pi * 10**2

    npt.assert_almost_equal(_ellipse_overlap(pt1, pt2, return_corr=True),
                            Aover / (np.pi * 10 * 20), decimal=3)


def test_min_merge_overlap():

    F = 2.0

    assert _min_merge_overlap(F) == 0.0

    F = 1.0

    npt.assert_almost_equal(_min_merge_overlap(F),
                            (2/3.) - np.sqrt(3)/(2*np.pi))

# Enable when packaging is added
# @pytest.mark.parameterize("pars",
#                           [((20, 30, 10, 5, 0), (20, 30, 10, 5, 0), 1)
#                            ])
def test_shell_similarity():

    t = np.linspace(0, 2 * np.pi, 50)

    test_cases = [((20, 30, 10, 5, 0), (20, 30, 10, 5, 0), 1),
                  ((20, 30, 10, 5, 0), (20, 30, 9, 5, 0), 1),
                  ((0, 0, 10, 10, 0), (0, 0, 10, 10, 0), 1),
                  ((0, 0, 10, 10, 0), (0, 0, 5, 5, 0), 0),
                  ((0, 0, 10, 10, 0), (5, 5, 5, 5, 0), 0.3),
                  ((0, 0, 10, 10, 0), (6, 6, 5, 5, 0), 0.24),
                  ((0, 0, 10, 10, 0), (7, 7, 5, 5, 0), 0.2)]

    def ellip_edges(t, pars):
        y, x, a, b, pa = pars
        return np.column_stack([x+a*np.cos(t), y+b*np.sin(t)])

    for params in test_cases:
        data1 = ellip_edges(t, params[0])
        data2 = ellip_edges(t, params[1])

        npt.assert_almost_equal(shell_similarity(data1, data2), params[2])
