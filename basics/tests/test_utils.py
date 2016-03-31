
import pytest
import numpy as np

from basics.utils import in_circle, in_ellipse


def test_in_circle():

    pt = (5, 5)
    pars = (0, 0, 10)

    assert in_circle(pt, pars)


def test_not_in_circle():

    pt = (5, 5)
    pars = (0, 0, 4)

    assert not in_circle(pt, pars)


# @pytest.mark.parameterize("pars",
#                           [(0, 0, 10, 10, 0),
#                            (0, 0, 10, 10, np.pi),
#                            (4, 4, 10, 5, 0),
#                            (4, 4, 10, 5, np.pi/4),
#                            (4, 4, 10, 5, np.pi/5),
#                            (4, 4, 10, 5, 1.23*np.pi)])
def test_in_ellipse():
    pt = (5, 5)

    for pars in [(0, 0, 10, 10, 0),
                 (0, 0, 10, 10, np.pi),
                 (4, 4, 10, 5, 0),
                 (4, 4, 10, 5, np.pi/4),
                 (4, 4, 10, 5, np.pi/5),
                 (4, 4, 10, 5, 1.23*np.pi)]:
        assert in_ellipse(pt, pars)


# @pytest.mark.parameterize("pars",
#                           [(0, 0, 4, 4, 0),
#                            (0, 0, 4, 4, np.pi),
#                            (-4, -4, 5, 3, 0),
#                            (-4, -4, 5, 3, np.pi/4),
#                            (-4, -4, 5, 3, np.pi/5),
#                            (-4, -4, 5, 3, 1.23*np.pi)])
def test_not_in_ellipse():
    pt = (5, 5)

    for pars in [(0, 0, 4, 4, 0),
                 (0, 0, 4, 4, np.pi),
                 (-4, -4, 5, 3, 0),
                 (-4, -4, 5, 3, np.pi/4),
                 (-4, -4, 5, 3, np.pi/5),
                 (-4, -4, 5, 3, 1.23*np.pi)]:
        assert not in_ellipse(pt, pars)
