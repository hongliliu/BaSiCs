
import numpy as np
from astropy.modeling.models import Ellipse2D
from astropy.stats import circvar
import astropy.units as u
import skimage.morphology as mo
from skimage.measure import approximate_polygon, find_contours
import scipy.ndimage as nd

from utils import ceil_int, eight_conn
from contour_orientation import shell_orientation
from polar_projection import reproject_image_into_polar


def find_bubble_edges(array, blob, max_extent=1.0,
                      nsig_thresh=1, value_thresh=None,
                      radius=None, return_mask=False, min_pixels=16,
                      filter_size=4, verbose=False,
                      min_radius_frac=0.5, try_local_bkg=True,
                      **kwargs):
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

        if try_local_bkg:
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
        bbox = Ellipse2D(True, 0.0, 0.0, major*max_extent, minor*max_extent,
                         pa).bounding_box

        y_range = ceil_int(bbox[0][1] - bbox[0][0] + 1 + filter_size)
        x_range = ceil_int(bbox[1][1] - bbox[1][0] + 1 + filter_size)

        extent_mask = np.zeros_like(array, dtype=bool)
        shell_thetas = []

        yy, xx = np.mgrid[-int(y_range / 2): int(y_range / 2) + 1,
                          -int(x_range / 2): int(x_range / 2) + 1]

        arr = array[max(0, y-int(y_range/2)):y+int(y_range/2)+1,
                    max(0, x-int(x_range/2)):x+int(x_range/2)+1]

        # import matplotlib.pyplot as p
        # p.subplot(121)
        # p.imshow(arr, origin='lower')
        # p.contour(arr <= value_thresh, colors='b')
        # p.subplot(122)
        # polar, r, theta = reproject_image_into_polar(arr)
        # p.imshow(polar, origin='lower')
        # p.contour(polar <= value_thresh, colors='b')
        # from skimage.filter import sobel_h
        # p.contour(sobel_h(polar), colors='r')
        # p.draw()
        # import time; time.sleep(0.1)
        # raw_input("?")
        # p.clf()

        # Adjust meshes if they exceed the array shape
        x_min = -min(0, x - int(x_range / 2))
        x_max = xx.shape[1] - max(0, x + int(x_range / 2) + 1 - array.shape[1])
        y_min = -min(0, y - int(y_range / 2))
        y_max = yy.shape[0] - max(0, y + int(y_range / 2) + 1 - array.shape[0])

        offset = (max(0, int(y - (y_range / 2))),
                  max(0, int(x - (x_range / 2))))

        yy = yy[y_min:y_max, x_min:x_max]
        xx = xx[y_min:y_max, x_min:x_max]

        dist_arr = np.sqrt(yy**2 + xx**2)

        smooth_mask = \
            _smooth_edges(arr <= value_thresh, filter_size, min_pixels)

        region_mask = \
            Ellipse2D(True, 0.0, 0.0, major*max_extent, minor*max_extent,
                      pa)(xx, yy).astype(bool)
        region_mask = nd.binary_dilation(region_mask, eight_conn, iterations=2)

        local_center = zip(*np.where(dist_arr == 0.0))[0]
        _make_bubble_mask(smooth_mask, local_center)

        # If the center is not contained within a bubble region, return
        # empties.
        if not smooth_mask.any() or smooth_mask.all():
            if return_mask:
                return np.array([]), 0.0, 0.0, value_thresh, smooth_mask

            return np.array([]), 0.0, 0.0, value_thresh

        orig_perim = find_contours(region_mask, 0, fully_connected='high')[0]
        # new_perim = find_contours(smooth_mask, 0, fully_connected='high')
        coords = []
        extent_mask = np.zeros_like(region_mask)
        # for perim in new_perim:
        #     perim = perim.astype(np.int)
        #     good_pts = \
        #         np.array([pos for pos, pt in enumerate(perim)
        #                   if region_mask[pt[0], pt[1]]])
        #     if not good_pts.any():
        #         continue

        #     # Now split into sections
        #     split_pts = consec_split(good_pts)

        #     # Remove the duplicated end point if it was initially connected
        #     if len(split_pts) > 1:
        #         # Join these if the end pts initially matched
        #         if split_pts[0][0] == split_pts[-1][-1]:
        #             split_pts[0] = np.append(split_pts[0],
        #                                      split_pts[-1][::-1])
        #             split_pts.pop(-1)

        #     for split in split_pts:
        #         coords.append(perim[split])

        #     extent_mask[perim[good_pts][:, 0], perim[good_pts][:, 1]] = True

        # Based on the curvature of the shell, only fit points whose
        # orientation matches the assumed centre.
        incoord, outcoord = shell_orientation(coords, local_center,
                                              verbose=False)


        # Now only keep the points that are not blocked from the centre pixel
        for pt in orig_perim:

            theta = np.arctan2(pt[0] - local_center[0],
                               pt[1] - local_center[1])

            num_pts = int(np.round(np.hypot(pt[0] - local_center[0],
                                            pt[1] - local_center[1]),
                                   decimals=0))

            y = np.round(np.linspace(local_center[0], pt[0], num_pts),
                         decimals=0).astype(np.int)

            x = np.round(np.linspace(local_center[1], pt[1], num_pts),
                         decimals=0).astype(np.int)

            dist = np.sqrt(y**2 + x**2)

            prof = smooth_mask[y, x]

            prof = prof[dist >= min_radius_frac * minor]
            y = y[dist >= min_radius_frac * minor]
            x = x[dist >= min_radius_frac * minor]

            # Look for the first 0 and ignore all others past it
            zeros = np.where(prof == 0)[0]

            # If none, move on
            if not zeros.any():
                continue

            edge = zeros[0]

            extent_mask[y[edge], x[edge]] = True
            coords.append((y[edge], x[edge]))
            shell_thetas.append(theta)

        # Sum the difference in angles (assuming points are in order)
        # arc_length = 0
        # for pts in coords:

        #     y, x = pts[:, 0], pts[:, 1]

        #     thetas = np.unwrap(np.arctan2(y - local_center[0],
        #                                   x - local_center[1]))

        #     arc_length += np.sum(np.diff(thetas))

        # Calculate the fraction of the region associated with a shell
        shell_frac = len(shell_thetas) / float(len(orig_perim))
        # shell_frac = arc_length / (2*np.pi)
        # coords = np.vstack(coords)
        # shell_thetas = np.arctan2(coords[:, 0] - local_center[0],
        #                           coords[:, 1] - local_center[1])

        shell_thetas = np.array(shell_thetas)
        coords = np.array(coords)

        # Use the theta values to find the standard deviation i.e. how
        # dispersed the shell locations are. Assumes a circle, but we only
        # consider moderately elongated ellipses, so the statistics approx.
        # hold.
        theta_var = np.sqrt(circvar(shell_thetas*u.rad)).value

        extent_coords = \
            np.vstack([pt + off for pt, off in
                       zip(np.where(extent_mask), offset)]).T

        if verbose:
            import matplotlib.pyplot as p
            ax = p.subplot(121)
            ax.imshow(arr, origin='lower',
                      interpolation='nearest')
            ax.contour(smooth_mask, colors='b')
            ax.contour(region_mask, colors='r')
            p.plot(coords[:, 1], coords[:, 0], 'bD')
            # for coord in coords:
            #     p.plot(coord[:, 1], coord[:, 0], 'bD')
            #     approx_coords = approximate_polygon(coord, np.sqrt(2))
            #     p.plot(approx_coords[:, 1], approx_coords[:, 0], 'mo')
            p.plot(local_center[1], local_center[0], 'gD')
            ax2 = p.subplot(122)
            ax2.imshow(extent_mask, origin='lower',
                       interpolation='nearest')
            import time; time.sleep(0.1)
            p.draw()
            raw_input("?")
            p.clf()

        if return_mask:
            return extent_coords, shell_frac, theta_var, value_thresh, \
                extent_mask

        return extent_coords, shell_frac, theta_var, value_thresh


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

    return bottom + 2 * sig, sig


def _smooth_edges(mask, filter_size, min_pixels):

    no_small = mo.remove_small_holes(mask, min_size=min_pixels,
                                     connectivity=2)

    open_close = \
        nd.binary_closing(nd.binary_opening(no_small, eight_conn), eight_conn)

    medianed = nd.median_filter(open_close, filter_size)

    return mo.remove_small_holes(medianed, min_size=min_pixels,
                                 connectivity=2)


def _make_bubble_mask(edge_mask, center):
    '''
    When a region is too large, unconnected and unrelated edges may be picked
    up. This removes those and only keeps the region that contains the center
    point.
    '''

    labels, num = nd.label(edge_mask)

    if num == 1:
        return edge_mask

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

        edge_mask[labels == n] = False


# Walking around the 3x3 neighbourhood starting at 0,0 and going clockwise
# Each is C-4 symmetric, so needs to be rotated 90 deg 3 more times than shown
connectivity = {1: [3, 4, 5, 6, 7], 2: [5, 6, 7]}
neighborhood = np.array([[3, 4, 5], [2, 0, 6], [1, 8, 7]])


def skeleton_gap_filling(mask):
    '''
    Fill single missing points in a skeleton using hit-or-miss transforms.
    '''

    hits = np.zeros_like(mask)

    for key in connectivity:
        element = neighborhood == key
        for posn in connectivity[key]:
            elem = np.logical_or(element, neighborhood == posn)
            hits += \
                nd.binary_hit_or_miss(mask, elem)
            print(elem.astype(int))
            for i in [1, 2, 3]:
                hits += \
                    nd.binary_hit_or_miss(mask, np.rot90(elem, k=i))
                print(np.rot90(elem, k=i).astype(int))
            raw_input("?")

    return mask + hits
