
import numpy as np
from astropy.modeling.models import Ellipse2D
from astropy.stats import circvar
import astropy.units as u

from profile import radial_profiles, _line_profile_coordinates
from utils import consec_split, find_nearest, floor_int, ceil_int


def find_bubble_edges(array, blob, max_extent=1.0,
                      nsig_thresh=1, value_thresh=None, min_radius_frac=0.5,
                      radius=None, return_mask=False, **kwargs):
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
        min_radius_frac : float, optional
            Sets a minimum distance to search for the bubble boundary. Defaults
            to 1/2 of the major radius.
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

        perimeter = 2 * np.pi * np.sqrt(0.5*(major**2 + minor**2))
        ntheta = ceil_int(1.5 * perimeter)

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

        dist_arr = np.sqrt(yy**2 + xx**2)

        centre = [arr[0] for arr in np.where(dist_arr == 0.0)]

        for vals in zip(*radial_profiles(array, blob,
                                         extend_factor=max_extent,
                                         append_end=True,
                                         ntheta=ntheta,
                                         return_thetas=True,
                                         **kwargs)):

            (dist, prof, end), theta = vals

            new_end = (max_extent * (end[0] - y),
                       max_extent * (end[1] - x))

            line_posns = []
            coords = [floor_int(coord)+cent for coord, cent in
                      zip(_line_profile_coordinates((0, 0), new_end), centre)]
            for coord in coords:
                line_posns.append(coord[dist_arr[coords] >=
                                  min_radius_frac*minor])

            prof = prof[dist >= min_radius_frac * minor]
            dist = dist[dist >= min_radius_frac * minor]

            above_thresh = np.where(prof >= value_thresh)[0]

            # This angle does not coincide with a shell.
            if above_thresh.size == 0:
                continue
            else:
                # Pick the end point on the first segment with > 2 pixels
                # above the threshold.
                segments = consec_split(above_thresh)

                for seg in segments:
                    if seg.size < 2:
                        continue

                    # Take the first position and use it to define the edge
                    dist_val = dist[seg[0]]

                    nearest_idx = \
                        find_nearest(dist_arr[line_posns],
                                     dist_val).astype(int)
                    end_posn = tuple([posn[nearest_idx] + off for
                                      posn, off in zip(line_posns, offset)])
                    extent_mask[end_posn] = True
                    shell_thetas.append(theta)
                    break
                else:
                    continue

        # Calculate the fraction of the region associated with a shell
        shell_frac = len(shell_thetas) / float(ntheta)

        # Use the theta values to find the standard deviation i.e. how
        # dispersed the shell locations are. Assumes a circle, but we only
        # consider moderately elongated ellipses, so the statistics approx.
        # hold.
        shell_thetas = np.array(shell_thetas)
        theta_var = np.sqrt(circvar(shell_thetas*u.rad)).value

        extent_coords = np.vstack(np.where(extent_mask)).T

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
