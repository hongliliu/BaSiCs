import numpy as np
from scipy import ndimage as ndi

'''
Copyright (C) 2011, the scikit-image team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
 3. Neither the name of skimage nor the names of its contributors may be
    used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''


def profile_line(img, src, dst, linewidth=1,
                 order=1, mode='constant', cval=0.0):
    """Return the intensity profile of an image measured along a scan line.

    Parameters
    ----------
    img : numeric array, shape (M, N[, C])
        The image, either grayscale (2D array) or multichannel
        (3D array, where the final axis contains the channel
        information).
    src : 2-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2-tuple of numeric scalar (float or int)
        The end point of the scan line.
    linewidth : int, optional
        Width of the scan, perpendicular to the line
    order : int in {0, 1, 2, 3, 4, 5}, optional
        The order of the spline interpolation to compute image values at
        non-integer coordinates. 0 means nearest-neighbor interpolation.
    mode : {'constant', 'nearest', 'reflect', 'mirror', 'wrap'}, optional
        How to compute any values falling outside of the image.
    cval : float, optional
        If `mode` is 'constant', what constant value to use outside the image.

    Returns
    -------
    return_value : array
        The intensity profile along the scan line. The length of the profile
        is the ceil of the computed length of the scan line.

    Examples
    --------
    >>> x = np.array([[1, 1, 1, 2, 2, 2]])
    >>> img = np.vstack([np.zeros_like(x), x, x, x, np.zeros_like(x)])
    >>> img
    array([[0, 0, 0, 0, 0, 0],
           [1, 1, 1, 2, 2, 2],
           [1, 1, 1, 2, 2, 2],
           [1, 1, 1, 2, 2, 2],
           [0, 0, 0, 0, 0, 0]])
    >>> profile_line(img, (2, 1), (2, 4))
    array([ 1.,  1.,  2.,  2.])

    Notes
    -----
    The destination point is included in the profile, in contrast to
    standard numpy indexing.
    """
    perp_lines = _line_profile_coordinates(src, dst, linewidth=linewidth)
    if img.ndim == 3:
        pixels = [ndi.map_coordinates(img[..., i], perp_lines,
                                      order=order, mode=mode, cval=cval)
                  for i in range(img.shape[2])]
        pixels = np.transpose(np.asarray(pixels), (1, 2, 0))
    else:
        pixels = ndi.map_coordinates(img, perp_lines,
                                     order=order, mode=mode, cval=cval)
    # intensities = pixels.mean(axis=1)
    intensities = np.nanmean(pixels, axis=1)

    # Return the distances from the source.
    dists = np.sqrt((perp_lines[0].mean(1) - src[0])**2 +
                    (perp_lines[1].mean(1) - src[1])**2)

    # Check for NaNs and remove those points
    if np.isnan(intensities).any():
        dists = dists[np.isfinite(intensities)]
        intensities = intensities[np.isfinite(intensities)]

    return intensities, dists


def _line_profile_coordinates(src, dst, linewidth=1):
    """Return the coordinates of the profile of an image along a scan line.

    Parameters
    ----------
    src : 2-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2-tuple of numeric scalar (float or int)
        The end point of the scan line.
    linewidth : int, optional
        Width of the scan, perpendicular to the line

    Returns
    -------
    coords : array, shape (2, N, C), float
        The coordinates of the profile along the scan line. The length of the
        profile is the ceil of the computed length of the scan line.

    Notes
    -----
    This is a utility method meant to be used internally by skimage functions.
    The destination point is included in the profile, in contrast to
    standard numpy indexing.
    """
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src
    theta = np.arctan2(d_row, d_col)

    length = np.ceil(np.hypot(d_row, d_col) + 1)
    # we add one above because we include the last point in the profile
    # (in contrast to standard numpy indexing)
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)

    # we subtract 1 from linewidth to change from pixel-counting
    # (make this line 3 pixels wide) to point distances (the
    # distance between pixel centers)
    col_width = (linewidth - 1) * np.sin(-theta) / 2
    row_width = (linewidth - 1) * np.cos(theta) / 2
    perp_rows = np.array([np.linspace(row_i - row_width, row_i + row_width,
                                      linewidth) for row_i in line_row])
    perp_cols = np.array([np.linspace(col_i - col_width, col_i + col_width,
                                      linewidth) for col_i in line_col])
    return np.array([perp_rows, perp_cols])


def radial_profiles(image, blob, ntheta=360, verbose=False,
                    extend_factor=1.5, append_end=False, return_thetas=False,
                    **kwargs):
    '''
    Calculate radial profiles from the centre of a bubble to its edge.

    Parameters
    ----------
    image : 2D np.ndarray
        Image to calculate the profiles from.
    blob : np.ndarray
        Contains the y, x, major radius, minor radius, and position angle of
        the blob.
    ntheta : int, optional
        Number of angles to compute the profile at.
    verbose : bool, optional
        Plots the profile and the positions in the image at each theta.
    extend_factor : float, optional
        Number of times past the major radius to compute the profile to.
    append_end : bool, optional
        Append the end point onto the returned list.
    return_thetas : bool, optional
        Return the array of theta values.

    Returns
    -------
    profiles : list
        Contains the distance and profile for each theta. It also contains the
        end point when append_end is enabled.
    thetas : np.ndarray, optional
        Returned when return_thetas is enabled.
    '''

    y0, x0, a, b, pa = blob.copy()[:5]

    a *= extend_factor
    b *= extend_factor

    thetas = np.linspace(0.0, 2*np.pi, ntheta)

    profiles = []

    for i, theta in enumerate(thetas):

        end_pt = (y0 + a*np.cos(theta)*np.sin(pa) + b*np.sin(theta)*np.cos(pa),
                  x0 + a*np.cos(theta)*np.cos(pa) - b*np.sin(theta)*np.sin(pa))

        profile, dists = profile_line(image, (y0, x0), end_pt, **kwargs)

        if append_end:
            profiles.append((dists, profile, end_pt))
        else:
            profiles.append((dists, profile))

        if verbose:
            import matplotlib.pyplot as p
            import time

            p.subplot(121)
            p.imshow(image, cmap='afmhot')
            p.plot(x0, y0, 'bD')
            p.plot(end_pt[1], end_pt[0], 'rD')
            p.xlim([0, image.shape[1]])
            p.ylim([0, image.shape[0]])

            p.subplot(122)
            p.plot(dists, profile, 'bD-')

            p.draw()
            time.sleep(0.05)

    if return_thetas:
        return profiles, thetas

    return profiles
