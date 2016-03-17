
import numpy as np
from astropy.io.fits import PrimaryHDU
from pvextractor import Path, extract_pv_slice


def pv_wedge(cube, center, length, min_theta, max_theta,
             ntheta=90, width=1):
    '''
    Create a PV slice from a wedge.
    '''

    y0, x0 = center

    thetas = np.linspace(min_theta, max_theta, ntheta)

    pv_slices = []

    for i, theta in enumerate(thetas):

        start_pt = (y0 - (length / 2.)*np.sin(theta),
                    x0 - (length / 2.)*np.cos(theta))

        end_pt = (y0 + (length / 2.)*np.sin(theta),
                  x0 + (length / 2.)*np.cos(theta))

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
