
import numpy as np
from astropy.modeling.models import Ellipse2D
from astropy.stats import circvar
import astropy.units as u
import skimage.morphology as mo
import scipy.ndimage as nd

from profile import radial_profiles, _line_profile_coordinates
from utils import consec_split, find_nearest, floor_int, ceil_int, eight_conn
from contour_orientation import shell_orientation


def find_bubble_edges(array, blob, max_extent=1.0,
                      nsig_thresh=1, value_thresh=None,
                      radius=None, return_mask=False, min_pixels=16,
                      filter_size=4, verbose=False, **kwargs):
        '''
        Expand/contract to match the contours in the data.

        Parameters
        ----------
        array : 2D numpy.ndarray or spectral_cube.LowerDimensionalObject
            Data used to define the region boundaries.
        max_extent : float, optional
            Multiplied by the major radius to set how far should be searched
            when searching for the boundary.
        nsig_thresh : float, optional
            Number of times sigma above the mean to set the boundary intensity
            requirement. This is used whenever the local background is higher
            than the given `value_thresh`.
        value_thresh : float, optional
            When given, sets the minimum intensity for defining a bubble edge.
            The natural choice is a few times the noise level in the cube.
        radius : float, optional
            Give an optional radius to use instead of the major radius defined
            for the bubble.
        kwargs : passed to profile.profile_line.

        Returns
        -------
        extent_coords : np.ndarray
            Array with the positions of edges.
        '''

        mean, std = intensity_props(array, blob)
        background_thresh = mean + nsig_thresh * std

        # Define a suitable background based on the intensity within the
        # elliptical region
        if value_thresh is None:
            value_thresh = background_thresh
        else:
            # If value_thresh is higher use it. Otherwise use the bkg.
            if value_thresh < background_thresh:
                value_thresh = background_thresh

        # Set the number of theta to be ~ the perimeter.

        y, x, major, minor, pa = blob[:5]

        # Use the ellipse model to define a bounding box for the mask.
        bbox = Ellipse2D(True, 0.0, 0.0, major, minor, pa).bounding_box

        y_range = ceil_int((max_extent * 2) *
                           (bbox[0][1] - bbox[0][0]))
        x_range = ceil_int((max_extent * 2) *
                           (bbox[1][1] - bbox[1][0]))

        offset = (int(y - (y_range / 2)), int(x - (x_range / 2)))

        extent_mask = np.zeros_like(array, dtype=bool)
        shell_thetas = []

        yy, xx = np.mgrid[-int(y_range / 2): int(y_range / 2) + 1,
                          -int(x_range / 2): int(x_range / 2) + 1]

        arr = array[max(0, y-int(y_range/2)):y+int(y_range/2)+1,
                    max(0, x-int(x_range/2)):x+int(x_range/2)+1]

        # Adjust meshes if they exceed the array shape
        x_min = -min(0, x - int(x_range / 2))
        x_max = xx.shape[1] - max(0, x + int(x_range / 2) + 1 - array.shape[1])
        y_min = -min(0, y - int(y_range / 2))
        y_max = yy.shape[0] - max(0, y + int(y_range / 2) + 1 - array.shape[0])

        yy = yy[y_min:y_max, x_min:x_max]
        xx = xx[y_min:y_max, x_min:x_max]

        smooth_mask = \
            _smooth_edges(arr > value_thresh, filter_size, min_pixels)

        region_mask = \
            Ellipse2D(True, 0.0, 0.0, major*max_extent, minor*max_extent,
                      pa)(yy, xx).astype(bool)

        local_center = (int(y_range)/2, int(x_range)/2)
        bubble_mask = _make_bubble_mask(smooth_mask, region_mask, local_center)

        # If the center is not contained within a bubble region, return
        # empties.
        if not bubble_mask.any():
            if return_mask:
                return np.array([]), 0.0, 0.0, bubble_mask

            return np.array([]), 0.0, 0.0

        orig_perim = perimeter_points(region_mask)
        new_perim = perimeter_points(bubble_mask)
        coords = np.array(list(set(new_perim) - set(orig_perim)))
        extent_mask = np.zeros_like(region_mask)
        extent_mask[coords[:, 0], coords[:, 1]] = True
        extent_mask = mo.medial_axis(extent_mask)

        # Based on the curvature of the shell, only fit points whose
        # orientation matches the assumed centre.
        incoord, outcoord = shell_orientation(extent_mask, local_center,
                                              verbose=False)

        shell_frac = np.sum(extent_mask) / float(len(coords))
        shell_thetas = np.arctan2(coords[:, 0], coords[:, 1])

        # Use the theta values to find the standard deviation i.e. how
        # dispersed the shell locations are. Assumes a circle, but we only
        # consider moderately elongated ellipses, so the statistics approx.
        # hold.
        theta_var = np.sqrt(circvar(shell_thetas*u.rad)).value

        if verbose:
            import matplotlib.pyplot as p
            ax = p.subplot(121)
            ax.imshow(bubble_mask, origin='lower',
                      interpolation='nearest')
            ax.contour(smooth_mask, colors='b')
            ax.contour(region_mask, colors='r')
            p.plot(coords[:, 1], coords[:, 0], 'bD')
            ax2 = p.subplot(122)
            ax2.imshow(extent_mask, origin='lower',
                       interpolation='nearest')
            p.show()

        extent_coords = \
            np.vstack([pt + off for pt, off in
                       zip(np.where(extent_mask), offset)]).T

        if return_mask:
            return extent_coords, shell_frac, theta_var, extent_mask

        return extent_coords, shell_frac, theta_var


def intensity_props(data, blob, min_rad=4):
    '''
    Return the mean and std for the elliptical region in the given data.

    Parameters
    ----------
    data : LowerDimensionalObject or SpectralCube
        Data to estimate the background from.
    blob : numpy.array
        Contains the properties of the region.

    '''

    y, x, major, minor, pa = blob[:5]

    inner_ellipse = \
        Ellipse2D(True, x, y, max(min_rad, 0.75*major),
                  max(min_rad, 0.75*minor), pa)

    yy, xx = np.mgrid[:data.shape[-2], :data.shape[-1]]

    ellip_mask = inner_ellipse(xx, yy).astype(bool)

    vals = data[ellip_mask]

    bottom = np.nanpercentile(vals, 2.5)
    fifteen = np.nanpercentile(vals, 15.)
    sig = fifteen - bottom

    return bottom + 2*sig, sig


def _smooth_edges(mask, filter_size, min_pixels):

    open_close = \
        nd.binary_closing(nd.binary_opening(mask, eight_conn), eight_conn)

    medianed = nd.median_filter(open_close, filter_size)

    return mo.remove_small_objects(medianed, min_size=min_pixels)


def perimeter_points(mask, method='erode'):
    if method is 'dilate':
        perim = np.logical_xor(nd.binary_dilation(mask, eight_conn), mask)
    elif method is 'erode':
        perim = np.logical_xor(mask, nd.binary_erosion(mask, eight_conn))
    else:
        raise TypeError("method must be 'erode' or 'dilate'.")
    return [(y, x) for y, x in zip(*np.where(perim))]


def _make_bubble_mask(edge_mask, region_mask, center):
    '''
    When a region is too large, unconnected and unrelated edges may be picked
    up. This removes those and only keeps the region that contains the center
    point.
    '''

    hole_regions = np.logical_and(region_mask, ~edge_mask)
    final_mask = np.logical_and(region_mask, edge_mask)

    labels, num = nd.label(hole_regions)

    if num == 1:
        return final_mask

    contains_center = 0

    for n in range(1, num+1):
        pts = zip(*np.where(labels == n))

        if center in pts:
            contains_center = n
            break

    if contains_center == 0:
        Warning("The center is not within any hole region.")

    for n in range(1, num + 1):
        if n == contains_center:
            continue

        final_mask[labels == n] = True

    return final_mask
