
from spectral_cube import SpectralCube
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
import os

from basics.bubble_segment3D import BubbleFinder

# data_path = "/media/eric/Data_3/LITTLE_THINGS/IC1613/"
data_path = "/media/eric/Data_3/LITTLE_THINGS/IC10/"
# data_path = "/home/ekoch/Data/IC10/"
# data_path = "/home/ekoch/Data/IC1613/"
# data_path = "/Users/eric/Data/"
# data_path = "/media/eric/Data_3/VLA/THINGS/HO_II/"

# cube = SpectralCube.read(os.path.join(data_path, "IC1613_NA_ICL001.fits"))
cube = SpectralCube.read(os.path.join(data_path, "IC10_NA_ICL001.fits"))
# cube = SpectralCube.read(os.path.join(data_path, "HO_II_NA_CUBE_THINGS.FITS"))

# Remove empty channels
# cube = cube[:, 500:1500, 500:1500]

galaxy_props = {"center_coord": SkyCoord("01h04m47.80", "+02d07m04.0", frame='icrs'),
                "inclination": Angle(37.9 * u.deg),  # Table 6 from Hunter+08
                "position_angle": Angle(71.0 * u.deg),  # Table 6 from Hunter+08
                "scale_height": 200. * u.pc}  # Assumed from Silich+06 (not well constrained)

bub_find = BubbleFinder(cube, keep_threshold_mask=True, distance=0.74 * u.Mpc,
                        galaxy_props=galaxy_props)

bub_find.get_bubbles(verbose=True, overlap_frac=0.5, multiprocess=True,
                     refit=False, nsig=1.5, min_corr=0.75, min_overlap=0.9,
                     min_channels=3, nprocesses=None)

# Show the moment 0
# mom0 = fits.getdata(os.path.join(data_path, "IC1613_NA_X0_P_R.fits")).squeeze()
mom0 = fits.getdata(os.path.join(data_path, "IC10_NA_X0_P_R.fits")).squeeze()
# mom0 = fits.getdata(os.path.join(data_path, "HO_II_NA_MOM0_THINGS.FITS")).squeeze()

bub_find.visualize_bubbles(moment0=mom0[500:1500, 500:1500])

catalog = bub_find.to_catalog()
