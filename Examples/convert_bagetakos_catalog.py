
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
from copy import copy

'''
Combine some columns and clean-up the Bagetakos table
'''

cat_path = "/media/eric/Data_3/VLA/THINGS/"

tab = Table.read(os.path.join(cat_path, "bagetakos_bubbles.txt"),
                 format='ascii')

# Mostly I want to combine the center coordinate columns since they're
# separated into RAh, RAm, etc...

ra = ["{}h{}m{}".format(hour, minu, sec) for hour, minu, sec in
      zip(tab["RAh"], tab["RAm"], tab["RAs"])]
dec = ["{}{}d{}m{}".format(sign if sign == "-" else "+", hour, minu, sec)
       for sign, hour, minu, sec in
       zip(tab["DE-"], tab["DEd"], tab["DEm"], tab["DEs"])]

coords = SkyCoord([(r, d) for r, d in zip(ra, dec)])

columns = copy(tab.colnames)

# Now remove the position columns
for name in ["RAh", "RAm", "RAs", "DE-", "DEd", "DEm", "DEs"]:
    columns.remove(name)

new_table = Table([tab[col] for col in columns] + [coords],
                  names=[tab[col].name for col in columns] + ["Coords"])

# Wrong units on the energy
new_table["log(E)"].unit = 1e50 * u.J

# Saving as an ecsv will read the coordinates back in as SkyCoords!
new_table.write(os.path.join(cat_path, "bagetakos_bubbles_reformatted.ecsv"),
                format='ascii.ecsv')
