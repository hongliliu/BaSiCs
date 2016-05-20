
import numpy as np
from astropy.io.fits import PrimaryHDU
from pvextractor import Path, extract_pv_slice
import scipy.ndimage as nd
from warnings import warn
from spectral_cube import SpectralCube


def pv_wedge(cube, center, length, min_theta, max_theta,
             ntheta=90, width=1):
    '''
    Create a PV slice from a wedge.
    '''

    y0, x0 = center

    thetas = np.linspace(min_theta, max_theta, ntheta)

    pv_slices = []

    for i, theta in enumerate(thetas):

        start_pt = (y0 - (length / 2.) * np.sin(theta),
                    x0 - (length / 2.) * np.cos(theta))

        end_pt = (y0 + (length / 2.) * np.sin(theta),
                  x0 + (length / 2.) * np.cos(theta))

        path = Path([start_pt, end_pt], width=width)

        if i == 0:
            pv_slice = extract_pv_slice(cube, path)
            pv_path_slice, header = pv_slice.data, pv_slice.header
        else:
            pv_path_slice = extract_pv_slice(cube, path).data

        pv_slices.append(pv_path_slice)

        # Track the smallest shape
        if i == 0:
            path_length = pv_path_slice.shape[1]
        else:
            new_path_length = pv_path_slice.shape[1]
            if new_path_length < path_length:
                path_length = new_path_length

    header["NAXIS1"] = path_length

    # Now loop through and average together
    avg_pvslice = np.zeros((cube.shape[0], path_length), dtype='float')

    for pvslice in pv_slices:
        avg_pvslice += pvslice[:, :path_length]

    avg_pvslice /= float(ntheta)

    return PrimaryHDU(avg_pvslice, header=header)


def warp_ellipse_to_circle(cube, a, b, pa, stop_if_huge=True):
    '''
    Warp a SpectralCube such that the given ellipse is a circle int the
    warped frame.

    Since you should **NOT** be doing this with a large cube, we're going
    to assume that the given cube is a subcube centered in the middle of the
    cube.

    This requires a rotation, then scaling. The equivalent matrix is:
    [b cos PA    b sin PA]
    [-a sin PA   a cos PA ].

    '''

    if cube._is_huge:
        if stop_if_huge:
            raise Warning("The cube has the huge flag enabled. Disable "
                          "'stop_if_huge' if you would like to continue "
                          "anyways with the warp.")
        else:
            warn("The cube has the huge flag enabled. This may use a lot "
                 "of memory!")

    # Let NaNs be 0
    data = cube.with_fill_value(0.0).filled_data[:].value

    warped_array = []

    for i in range(cube.shape[0]):
        warped_array.append(nd.zoom(nd.rotate(data[i], np.rad2deg(-pa)),
                                    (1, a / b)))

    warped_array = np.array(warped_array)

    # We want to mask outside of the original bounds
    mask = np.ones(data.shape[1:])
    warp_mask = \
        np.isclose(nd.zoom(nd.rotate(mask, np.rad2deg(-pa)),
                           (1, a / b)), 1)

    # There's probably a clever way to transform the WCS, but all the
    # solutions appear to need pyast/starlink. The output of the wrap should
    # give a radius of b and the spectral dimension is unaffected.
    # Also this is hidden and users won't be able to use this weird cube
    # directly
    warped_cube = SpectralCube(warped_array * cube.unit, cube.wcs)
    warped_cube = warped_cube.with_mask(warp_mask)

    return warped_cube
