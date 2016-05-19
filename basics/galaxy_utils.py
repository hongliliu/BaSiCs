
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord


def galactic_radius(coord, gal_center, distance, pa, inc):

    if not isinstance(coord, SkyCoord):
        raise TypeError("coord must be a SkyCoord")

    if not isinstance(gal_center, SkyCoord):
        raise TypeError("gal_center must be a SkyCoord")

    if not distance.unit.is_equivalent(u.pc):
        raise TypeError("distance must have an appropriate unit of"
                        " distance.")

    PA = gal_center.position_angle(coord)
    GalPA = PA - pa
    GCDist = coord.separation(gal_center)

    # Transform into galaxy plane
    Rplane = distance * np.tan(GCDist)
    Xplane = Rplane * np.cos(GalPA)
    Yplane = Rplane * np.sin(GalPA)

    Xgal = Xplane
    Ygal = Yplane / np.cos(inc)
    Rgal = np.sqrt(Xgal**2 + Ygal**2)

    return Rgal
