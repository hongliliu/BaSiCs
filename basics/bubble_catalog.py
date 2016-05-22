
import numpy as np
from astropy.table import Table, Column
import astropy.units as u
from astropy.coordinates import SkyCoord

from galaxy_utils import gal_props_checker


class PP_Catalog(object):
    """docstring for PP_Catalog"""
    def __init__(self, bubbles):
        super(PP_Catalog, self).__init__()
        self.bubbles = bubbles


class PPV_Catalog(object):
    """
    Return a table structure of the bubble properties and make it easy to
    visualize the populations.

    Parameters
    ----------
    bubbles : list of Bubble3D objects or astropy.table.Table
        Bubble3D objects.
    """
    def __init__(self, bubbles, galaxy_props=None):
        super(PPV_Catalog, self).__init__()

        if isinstance(bubbles, list):
            if galaxy_props is None:
                raise TypeError("Galaxy properties must be given when bubbles"
                                " is a list of bubble objects.")
            self.table = self.create_table(bubbles, galaxy_props)

        elif isinstance(bubbles, Table):
            self.table = bubbles
        else:
            raise TypeError("bubbles must be a list of bubble objects or a "
                            "pre-made astropy table.")

    @staticmethod
    def from_file(filename, format='ascii.ecsv'):

        try:
            tab = Table.read(filename, format=format)
        except Exception as e:
            # Add something a more helpful here.
            raise e

        self = PPV_Catalog(tab)

        return self

    def create_table(self, bubbles, galaxy_props):
        '''
        Create a Table from a list of bubbles
        '''

        # Check to make sure all of the properties are there
        gal_props_checker(galaxy_props)

        # Create columns of the bubble properties
        props = {"pa": [u.deg, "Position angle of the bubble"],
                 "bubble_type": [u.dimensionless_unscaled, "Type of bubble"],
                 "velocity_center": [u.km / u.s, "Center velocity"],
                 "velocity_width": [u.km / u.s, "Range of velocities bubble"
                                                " is detected."],
                 "eccentricity": [u.dimensionless_unscaled,
                                  "Shape eccentricity"],
                 "expansion_velocity": [u.km / u.s, "Expansion velocity"],
                 "avg_shell_flux_density": [u.K * u.km / u.s,
                                            "Average flux density in bubble "
                                            "shell"],
                 "total_shell_flux_density": [u.K * u.km / u.s,
                                              "Total flux density in bubble "
                                              "shell"],
                 "shell_column_density": [u.cm ** -2,
                                          "Average column density in the "
                                          "shell"],
                 "hole_contrast": [u.dimensionless_unscaled,
                                   "Average intensity difference between hole"
                                   " and shell."],
                 "diameter_physical": [u.pc, "Physical diameter"],
                 "major_physical": [u.pc, "Physical major radius"],
                 "minor_physical": [u.pc, "Physical minor radius"],
                 "diameter_angular": [u.deg, "Angular diameter"],
                 "major_angular": [u.deg, "Angular major radius"],
                 "minor_angular": [u.deg, "Angular minor radius"],
                 "galactic_radius": [u.kpc, "Galactic radius of the"
                                            " center."],
                 "galactic_pa": [u.deg, "Galactic PA of the center."]}

        prop_funcs = {"tkin": [u.Myr, "Kinetic age of the bubble.", {}],
                      "shell_volume_density":
                      [u.cm ** -3,
                       "Average hydrogen volume "
                       "density in the shell.",
                       {"scale_height": galaxy_props["scale_height"],
                        "inclination": galaxy_props["inclination"]}],
                      "volume":
                      [u.pc ** 3, "Volume of the hole.",
                       {"scale_height": galaxy_props["scale_height"]}],
                      "hole_mass":
                      [u.Msun, "Inferred mass of the hole from the shell"
                               " volume density.",
                       {"scale_height": galaxy_props["scale_height"],
                        "inclination": galaxy_props["inclination"]}],
                      "formation_energy":
                      [u.erg, "Energy required to create the hole.",
                       {"scale_height": galaxy_props["scale_height"],
                        "inclination": galaxy_props["inclination"]}]}

        columns = []

        # The center coordinates are different, since they're SkyCoords
        columns.append(SkyCoord([bub.center_coordinate for bub in bubbles]))

        # Same for is_closed
        columns.append(Column([bub.is_closed for bub in bubbles],
                              unit=u.dimensionless_unscaled,
                              description="Closed or partial shell.",
                              name="closed_shell"))

        # Add the properties
        for name in props:
            unit, descrip = props[name]
            columns.append(Column([getattr(bub, name).to(unit).value for bub in
                                   bubbles],
                                  name=name, description=descrip,
                                  unit=unit.to_string()))

        # Add the functions
        for name in prop_funcs:
            unit, descrip, imps = prop_funcs[name]
            columns.append(Column([getattr(bub, name)(**imps).to(unit).value
                                   for bub in bubbles], name=name,
                                  description=descrip,
                                  unit=unit.to_string()))

        self.table = Table(columns)

    def population_statistics(self, percentiles=[25, 50, 75]):
        '''
        Return percentiles of properties in the population
        '''
        pass

    def histogram_parameters(self, show_params=None):
        pass

    def scatter_parameters(self, show_params=None):
        pass

    def triangle_plot(self, show_params=None, **kwargs):

        try:
            from triangle import cornerplot
        except ImportError:
            raise ImportError("The triangle package must be installed.")

        pass

    def write_table(self, filename, format='ascii.ecsv'):
        '''
        Write the table. Format must be supported by astropy.table.
        '''
        self.table.write(filename, format=format)
