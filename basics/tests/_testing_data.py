
from astropy.modeling.models import Gaussian2D, Ellipse2D
import numpy as np
from spectral_cube.lower_dimensional_structures import Projection
from astropy import wcs
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.utils import ellipse_extent
from scipy.special import lambertw

from basics.log import overlap_metric

# Set the seed
np.random.seed(535463254)


def shell_model(major=50, minor=40, ratio=1.25, amp=0.5, pa=0, xcent=0,
                ycent=0, ring_large=None, ring_small=[40, 32]):
    '''
    Difference of Gaussians with the middle clipped to the background.
    '''

    gauss_model = \
        Gaussian2D(shell_max(amp, minor, ratio),
                   xcent, ycent, ratio*major, ratio*minor, theta=pa) - \
        Gaussian2D(shell_max(amp, minor, ratio),
                   xcent, ycent, major, minor, theta=pa)

    if ring_large is None and ring_small is None:
        return gauss_model
    elif ring_small is None and ring_large is not None:
        ring = Ellipse2D(True, xcent, ycent, ring_large[0], ring_large[1], pa)
    elif ring_small is not None and ring_large is None:
        ring = InvertedEllipse2D(True, xcent, ycent, ring_small[0],
                                 ring_small[1], pa)
    else:
        ring = \
            Ellipse2D(True, xcent, ycent, ring_large[0], ring_large[1], pa) - \
            Ellipse2D(True, xcent, ycent, ring_small[0], ring_small[1], pa)

    return gauss_model * ring


def shell_max(amplitude, radius, ratio):
    '''
    Compute the amplitude of the gaussians given the requested
    amplitude of the shell.
    '''

    small_rad = radius
    large_rad = ratio * radius

    # Solve for the position of the maximum. Verified with Wolfram Alpha.
    lamb_arg = (small_rad*large_rad)**2 / (large_rad**2 - small_rad**2)
    x = 0.5*(small_rad * large_rad) * \
        np.sqrt(lambertw(lamb_arg).real / (large_rad**2 - small_rad**2))

    func_val = np.exp(-0.5 * x**2 / large_rad**2) - \
        np.exp(-0.5 * x**2 / small_rad**2)

    return amplitude / func_val


def add_holes(shape, nholes=100, rad_max=30, rad_min=10, hole_level=None,
              return_info=False, max_corr=0.5, max_trials=500, max_eccent=2.):
    ylim, xlim = shape

    array = np.ones(shape) * 255

    yy, xx = np.mgrid[-ylim/2:ylim/2, -xlim/2:xlim/2]

    params = np.empty((0, 5))

    ct = 0
    trial = 0
    while True:

        xcenter = np.random.random_integers(-(xlim/2-rad_max),
                                            high=xlim/2-rad_max)
        ycenter = np.random.random_integers(-(ylim/2-rad_max),
                                            high=ylim/2-rad_max)
        major = np.random.uniform(rad_min, rad_max)
        minor = np.random.uniform(major / float(max_eccent), major)
        pa = np.random.uniform(0, np.pi)

        blob = np.array([ycenter, xcenter, major, minor, pa])

        if ct != 0:
            corrs = np.array([overlap_metric(par, blob, return_corr=True)
                              for par in params])
            if np.any(corrs > max_corr):
                trial += 1
                continue

        if hole_level is None:
            # Choose random hole bkg levels
            bkg = np.random.random_integers(0, 230)
        else:
            bkg = hole_level

        ellip = Ellipse2D(True, xcenter, ycenter, major, minor, pa)

        array[ellip(yy, xx).astype(bool)] = bkg

        params = np.append(params, blob[np.newaxis, :], axis=0)

        ct += 1
        trial += 1

        if ct == nholes:
            break

        if trial == max_trials:
            Warning("Only found %i valid regions in %i trials." %
                    (ct, max_trials))
            break

    if return_info:
        return array, params

    return array


def add_gaussian_peaks(array, nholes=100, rad_max=30, rad_min=10):
    ylim, xlim = array.shape

    yy, xx = np.mgrid[-ylim/2:ylim/2, -xlim/2:xlim/2]

    xcenters = np.random.random_integers(-(xlim/2-rad_max),
                                         high=xlim/2-rad_max, size=nholes)
    ycenters = np.random.random_integers(-(ylim/2-rad_max),
                                         high=ylim/2-rad_max, size=nholes)
    radii = np.random.random_integers(rad_min, rad_max, size=nholes)

    for x, y, radius in zip(xcenters, ycenters, radii):
        amp = np.random.uniform(0.5, 0.75)

        gauss = Gaussian2D.evaluate(yy, xx,
                                    amp, x, y, radius, radius, 0.0)

        array += gauss

    return array


def add_gaussian_holes(array, nholes=100, rad_max=30, rad_min=10,
                       return_info=False, hole_level=None):
    ylim, xlim = array.shape

    yy, xx = np.mgrid[-ylim/2:ylim/2, -xlim/2:xlim/2]

    xcenters = np.random.random_integers(-(xlim/2-rad_max),
                                         high=xlim/2-rad_max, size=nholes)
    ycenters = np.random.random_integers(-(ylim/2-rad_max),
                                         high=ylim/2-rad_max, size=nholes)
    radii = np.random.random_integers(rad_min, rad_max, size=nholes)

    if hole_level is None:
        # Choose random hole bkg levels
        bkgs = np.random.uniform(0.5, 0.75, size=nholes)
    else:
        bkgs = np.array([hole_level] * nholes, dtype=np.int)

    for x, y, radius, amp in zip(xcenters, ycenters, radii, bkgs):

        gauss = Gaussian2D.evaluate(yy, xx,
                                    amp, x, y, radius, radius, 0.0)

        array -= gauss

    if return_info:
        return array, np.vstack([ycenters+ylim/2, xcenters+xlim/2, radii])

    return array


class InvertedEllipse2D(Fittable2DModel):
    """
    A 2D Ellipse model.

    Parameters
    ----------
    amplitude : float
        Value of the ellipse.

    x_0 : float
        x position of the center of the disk.

    y_0 : float
        y position of the center of the disk.

    a : float
        The length of the semimajor axis.

    b : float
        The length of the semiminor axis.

    theta : float
        The rotation angle in radians of the semimajor axis.  The
        rotation angle increases counterclockwise from the positive x
        axis.

    See Also
    --------
    Disk2D, Box2D

    Notes
    -----
    Model formula:

    .. math::

        f(x, y) = \\left \\{
                    \\begin{array}{ll}
                      \\mathrm{amplitude} & : \\left[\\frac{(x - x_0) \\cos
                        \\theta + (y - y_0) \\sin \\theta}{a}\\right]^2 +
                        \\left[\\frac{-(x - x_0) \\sin \\theta + (y - y_0)
                        \\cos \\theta}{b}\\right]^2  \\leq 1 \\\\
                      0 & : \\mathrm{otherwise}
                    \\end{array}
                  \\right.

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        from astropy.modeling.models import Ellipse2D
        from astropy.coordinates import Angle
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        x0, y0 = 25, 25
        a, b = 20, 10
        theta = Angle(30, 'deg')
        e = Ellipse2D(amplitude=100., x_0=x0, y_0=y0, a=a, b=b,
                      theta=theta.radian)
        y, x = np.mgrid[0:50, 0:50]
        fig, ax = plt.subplots(1, 1)
        ax.imshow(e(x, y), origin='lower', interpolation='none', cmap='Greys_r')
        e2 = mpatches.Ellipse((x0, y0), 2*a, 2*b, theta.degree, edgecolor='red',
                              facecolor='none')
        ax.add_patch(e2)
        plt.show()
    """

    amplitude = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    a = Parameter(default=1)
    b = Parameter(default=1)
    theta = Parameter(default=0)

    @staticmethod
    def evaluate(x, y, amplitude, x_0, y_0, a, b, theta):
        """Two dimensional Ellipse model function."""

        xx = x - x_0
        yy = y - y_0
        cost = np.cos(theta)
        sint = np.sin(theta)
        numerator1 = (xx * cost) + (yy * sint)
        numerator2 = -(xx * sint) + (yy * cost)
        in_ellipse = (((numerator1 / a) ** 2 + (numerator2 / b) ** 2) <= 1.)
        return np.select([~in_ellipse], [amplitude])

    @property
    def bounding_box(self):
        """
        Tuple defining the default ``bounding_box`` limits.

        ``((y_low, y_high), (x_low, x_high))``
        """

        a = self.a
        b = self.b
        theta = self.theta.value
        dx, dy = ellipse_extent(a, b, theta)

        return ((self.y_0 - dy, self.y_0 + dy),
                (self.x_0 - dx, self.x_0 + dx))
