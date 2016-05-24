
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, Angle

GALAXY_KEYS = ["inclination", "position_angle", "center_coord",
               "scale_height"]


def galactic_radius_pa(coord, gal_center, distance, pa, inc, out_unit=u.pc):

    if not isinstance(coord, SkyCoord):
        raise TypeError("coord must be a SkyCoord")

    if not isinstance(gal_center, SkyCoord):
        raise TypeError("gal_center must be a SkyCoord")

    if not distance.unit.is_equivalent(u.pc):
        raise TypeError("distance must have an appropriate unit of"
                        " distance.")

    if not out_unit.is_equivalent(u.pc):
        raise TypeError("out_unit must have an appropriate unit of"
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
    Rgal = np.sqrt(Xgal**2 + Ygal**2).to(out_unit)

    return Rgal, GalPA


def gal_props_checker(input_dict):
    '''
    Ensure the dictionary passed has all of the galactic parameters needed
    set.
    '''

    # Make sure all of the kwargs are given.
    in_input = [True if key in GALAXY_KEYS else False
                for key in input_dict]
    if not np.all(in_input):
        missing = list(set(GALAXY_KEYS) - set(in_input))
        raise KeyError("galaxy_props is missing these keys: {}"
                       .format(missing))

    if not isinstance(input_dict["center_coord"], SkyCoord):
        raise TypeError("center_coords must be a SkyCoord.")

    if not input_dict["scale_height"].unit.is_equivalent(u.pc):
        raise u.UnitsError("scale_height must have a unit of distance")

    if not isinstance(input_dict["inclination"], u.Quantity):
        raise TypeError("inclination must be an Quantity.")
        if not input_dict["inclination"].unit.is_equivalent(u.deg):
            raise u.UnitsError("inclination must have an angular unit.")

    if not isinstance(input_dict["position_angle"], u.Quantity):
        raise TypeError("position_angle must be an Quantity.")
        if not input_dict["position_angle"].unit.is_equivalent(u.deg):
            raise u.UnitsError("position_angle must have an angular unit.")
