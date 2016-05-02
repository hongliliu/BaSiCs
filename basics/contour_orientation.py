
import numpy as np
from skimage.measure import subdivide_polygon, approximate_polygon
import scipy.ndimage as nd


def shell_orientation(coords, center=None, diff_thresh=0.5, smooth_width=5,
                      verbose=False, interactive=True):
    '''
    Reject coordinate points if their local curvature does not point
    to the center of the ellipse.
    '''

    breaks = np.empty((0, 2))

    new_coords = []

    # Loop through each segment, highlighting which points belong to the
    # region.
    for coord in coords:

        # Is it a closed contour?
        if (coord[0] == coord[-1]).all():
            is_closed = True
        else:
            is_closed = False

        # Approx w/ splines
        # coords = subdivide_polygon(coords, preserve_ends=True)
        # coords = approximate_polygon(coords, tolerance=2)

        num_diff = 2

        # Can't calculate with < 3 points.
        if len(coord) < 2 * num_diff + 1:
            continue

        y, x = coord.T.copy()

        if center is None:
            ymean = y.mean()
            xmean = x.mean()
            y -= ymean
            x -= xmean
        else:
            y -= center[0]
            x -= center[1]

        if smooth_width is not None:
            if smooth_width < 1.0:
                smooth_width = max(3, smooth_width * len(coord))

            if is_closed:
                mode = 'wrap'
            else:
                mode = 'mirror'

            y = nd.gaussian_filter1d(y.copy(), smooth_width, mode=mode)
            x = nd.gaussian_filter1d(x.copy(), smooth_width, mode=mode)

            if is_closed:
                y[-1] = y[0]
                x[-1] = x[0]

        yprime = richardson_diff(y)
        xprime = richardson_diff(x)

        theta = np.unwrap(np.arctan2(yprime, xprime))

        thetaprime = richardson_diff(theta)

        curvature = thetaprime / np.sqrt(yprime**2 + xprime**2)

        break_idx = np.where(np.abs(curvature) > diff_thresh)[0]

        if center is None:
            pts = np.vstack([y[break_idx] + ymean, x[break_idx] + xmean]).T
        else:
            pts = np.vstack([y[break_idx] + center[0],
                             x[break_idx] + center[1]]).T

        breaks = np.vstack([breaks, pts])

        prev_idx = 0
        for idx in break_idx:
            new_split = coord[prev_idx:idx]
            prev_idx = idx
            if len(new_split) < 3:
                continue
            new_coords.append(new_split)

        if verbose:
            import matplotlib.pyplot as p

            p.subplot(131)
            p.plot(theta, 'bD-')
            p.subplot(132)
            p.plot(curvature, 'bD-')
            p.subplot(133)
            p.plot(x, y, 'bD-')
            p.plot(x[np.c_[0, -1]], y[np.c_[0, -1]], 'rD')
            p.plot(x[np.c_[break_idx]], y[np.c_[break_idx]], 'kD')
            p.plot(0, 0, 'gD')

            if interactive:
                p.draw()
                raw_input("?")
                p.clf()
            else:
                p.show()

    breaks = np.array(breaks)

    return new_coords, breaks


def circle_center(pt1, pt2, pt3, cent=None):
    '''
    Find the center of the circle through 3 points.
    '''

    norm12 = np.linalg.norm(pt1-pt2)
    norm13 = np.linalg.norm(pt1-pt3)
    norm23 = np.linalg.norm(pt3-pt2)

    b1 = norm23**2 * (norm13**2 + norm12**2 - norm23**2)
    b2 = norm13**2 * (norm23**2 + norm12**2 - norm13**2)
    b3 = norm12**2 * (norm23**2 + norm13**2 - norm12**2)

    # Check for straight segments
    if b1 + b2 + b3 == 0.0:
        if cent is None:
            return np.NaN

        seg_theta = np.arctan2(pt3[1] - pt1[1], pt3[0] - pt1[0])

        theta_cent = np.arctan2(cent[1] - pt2[1], cent[0] - pt2[0])

        norm_thetas = \
            np.unwrap(np.array([seg_theta - 0.5*np.pi, seg_theta + 0.5*np.pi]),
                      discont=-np.pi)

        theta_diff = np.abs(np.arctan2(np.sin(norm_thetas - theta_cent),
                                       np.cos(norm_thetas - theta_cent)))

        theta_norm = norm_thetas[np.argmin(theta_diff)]

        R = np.sqrt((pt2[0] - cent[0])**2 + (pt2[1] - cent[1])**2)

        P = (pt2[0] + R*np.cos(theta_norm), pt2[1] + R*np.sin(theta_norm))

    else:
        P = np.vstack([pt1, pt2, pt3]).T.dot(np.vstack([b1, b2, b3]))
        P /= b1 + b2 + b3

    return P


def richardson_diff(pts, is_closed=False):
    '''
    Richardson difference for approximating the derivative.

    Method from:
    http://iopscience.iop.org/article/10.1088/0957-0233/16/9/007/meta
    '''

    if len(pts) < 5:
        raise IndexError("Cannot compute with less than 5 points.")

    if pts[-1] == pts[0]:
        is_closed = True

    if is_closed:
        numer = [pts[(t + 2) % len(pts)] - pts[t - 2] +
                 8 * (pts[(t + 1) % len(pts)] - pts[t - 1])
                 for t in range(len(pts))]
    else:
        numer = [pts[t + 2] - pts[t - 2] + 8 * (pts[t + 1] - pts[t - 1])
                 for t in range(2, len(pts) - 2)]

        # Append an extra two points on either side, assuming that the
        # difference for the end points is the same as the one before/after

        numer = [numer[0]] * 2 + numer
        numer = numer + [numer[-1]] * 2

    return np.array(numer) / 12.


def contour_breaks(coords, ntheta=180, prob_thresh=0.5):
    '''
    Find break points in contours using the rotation method of Shen et al.(2010)
    http://dx.doi.org/10.1016/S0167-8655(99)00130-0
    '''

    thetas = np.linspace(0., 2 * np.pi, ntheta)

    for coord in coords:

        is_closed = False
        if (coord[-1] == coord[0]).all():
            is_closed = True
            coord = coord.copy()[: -1]

        # coord = subdivide_polygon(coord.copy(), preserve_ends=True)

        prob_break = np.zeros_like(coord[:, 0], dtype=np.float)

        coord = coord.copy() - coord.mean(0)

        for theta in thetas:
            ang_coord = rotate_points(coord, theta)

            if is_closed:
                breaks = [is_break_point(ang_coord[i - 1], ang_coord[i],
                                         ang_coord[(i + 1) % len(coord)])
                          for i in range(len(coord))]
            else:
                # Set the first and last points to False by default
                breaks = [False]
                breaks.extend([is_break_point(ang_coord[i - 1], ang_coord[i],
                                              ang_coord[i + 1])
                               for i in range(1, len(coord) - 1)])
                breaks.append(False)

            breaks = np.array(breaks).astype(np.int8)

            prob_break += breaks

        prob_break /= float(ntheta)

        coord_breaks = np.where(prob_break > prob_thresh)[0]

        import matplotlib.pyplot as p

        p.subplot(121)
        p.plot(coord[:, 1], coord[:, 0], 'bD-')
        p.plot(coord[coord_breaks, 1], coord[coord_breaks, 0], 'ro')
        p.plot(coord[:, 1][np.c_[0, -1]], coord[:, 0][np.c_[0, -1]], 'gD')
        p.subplot(122)
        p.plot(prob_break)
        p.draw()
        import time
        time.sleep(0.1)
        raw_input("?")
        p.clf()


def rotate_points(coord, theta):

    y_rot = coord[:, 0] * np.sin(theta)
    x_rot = coord[:, 1] * np.cos(theta)

    return np.vstack([y_rot, x_rot]).T


def is_break_point(pt1, pt2, pt3):
    y1, x1 = pt1
    y2, x2 = pt2
    y3, x3 = pt3
    cond1 = np.logical_and(x1 > x2, x2 < x3)
    cond2 = np.logical_and(x1 < x2, x2 > x3)
    cond3 = np.logical_and(y1 > y2, y2 < y3)
    cond4 = np.logical_and(y1 < y2, y2 > y3)

    return np.logical_or(np.logical_or(cond1, cond2),
                         np.logical_or(cond3, cond4))
