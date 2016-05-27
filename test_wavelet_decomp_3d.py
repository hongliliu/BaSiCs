
from spectral_cube import SpectralCube
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import os

from basics.bubble_segment3D import BubbleFinder

# data_path = "/media/eric/Data_3/LITTLE_THINGS/IC1613/"
# data_path = "/media/eric/Data_3/LITTLE_THINGS/IC10/"
# data_path = "/home/ekoch/Data/IC10/"
# data_path = "/home/ekoch/Data/IC1613/"
# data_path = "/Users/eric/Data/"
data_path = "/media/eric/Data_3/VLA/THINGS/HO_II/"
# data_path = "/media/eric/Data_3/VLA/THINGS/NGC_2403/"

# cube = SpectralCube.read(os.path.join(data_path, "IC1613_NA_ICL001.fits"))
# cube = SpectralCube.read(os.path.join(data_path, "IC10_NA_ICL001.fits"))
cube = SpectralCube.read(os.path.join(data_path, "HO_II_NA_CUBE_THINGS.FITS"))
# cube = SpectralCube.read(os.path.join(data_path, "NGC_2403_NA_CUBE_THINGS.FITS"))


# IC 1613
# galaxy_props = {"center_coord": SkyCoord("01h04m47.80", "+02d07m04.0",
#                                          frame='icrs'),
#                 "inclination": 37.9 * u.deg,  # Table 6 from Hunter+08
#                 "position_angle": 71.0 * u.deg,  # Table 6 from Hunter+08
#                 "scale_height": 200. * u.pc}  # Assumed from Silich+06 (not well constrained)
# distance = 0.74 * u.Mpc
# slicer = (slice(None), slice(500, 1500), slice(500, 1500))

# HO II
galaxy_props = {"center_coord": SkyCoord("08h19m05.0", "+70d43m12.0",
                                         frame='icrs'),
                # Table 1 from Walter+08
                "inclination": 41 * u.deg,
                # Table 1 from Walter+08
                "position_angle": 177.0 * u.deg,
                "scale_height": 340. * u.pc}  # Bagetakos Table 3
distance = 3.4 * u.Mpc
slicer = (slice(None), slice(None), slice(None))

# Apply the rough slice
cube = cube[slicer]


bub_find = BubbleFinder(cube, keep_threshold_mask=True, distance=distance,
                        galaxy_props=galaxy_props)

bub_find.get_bubbles(verbose=True, overlap_frac=0.5, multiprocess=True,
                     refit=False, nsig=1.5, min_corr=0.7, min_overlap=0.8,
                     global_corr=0.5, min_channels=3, nprocesses=None)

# bub_find.recluster("saved_bubbles", verbose=True, multiprocess=True,
#                    refit=False, nsig=1.5, min_corr=0.7, min_overlap=0.8,
#                    global_corr=0.7,
#                    min_channels=3, nprocesses=None)

# Show the moment 0
# mom0 = fits.getdata(os.path.join(data_path, "IC1613_NA_X0_P_R.fits")).squeeze()
# mom0 = fits.getdata(os.path.join(data_path, "IC10_NA_X0_P_R.fits")).squeeze()
mom0 = fits.getdata(os.path.join(data_path, "HO_II_NA_MOM0_THINGS.FITS")).squeeze()
# mom0 = fits.getdata(os.path.join(data_path, "NGC_2403_NA_MOM0_THINGS.FITS")).squeeze()

bub_find.visualize_bubbles(moment0=mom0[slicer[1:]], plot_twoD_shapes=True)

catalog = bub_find.to_catalog()
