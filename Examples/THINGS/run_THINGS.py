
import os
import sys
import numpy as np
from astropy.io import fits
from spectral_cube import SpectralCube
import matplotlib.pyplot as p
from basics import BubbleFinder

# Get the galaxy properties
# from info_THINGS import galaxy_props
execfile("/lustre/home/ekoch/code_repos/BaSiCs/Examples/THINGS/info_THINGS.py")

# Path to data. The folders should have the same names
datapath = sys.argv[1]

# The folder names should also be the object name
name = datapath.split("/")[-1]

galaxy_prop = galaxy_props[name]

cube = SpectralCube.read(os.path.join(datapath,
                                      "{}_NA_CUBE_THINGS.FITS".format(name)))
mom2 = fits.getdata(os.path.join(datapath,
                                 "{}_NA_MOM2_THINGS.FITS".format(name)))
mom0 = fits.getdata(os.path.join(datapath,
                                 "{}_NA_MOM0_THINGS.FITS".format(name)))

mom0 = mom0.squeeze()
lwidth = np.sqrt(mom2.squeeze())

# The first channel is used to estimate the noise level. I think this is an
# ok choice for the whole set, though the continuum sources in NGC 3031
# might be an issue.
bub_find = BubbleFinder(cube, distance=galaxy_prop["distance"],
                        galaxy_props=galaxy_prop)

# NOTE: overlap_frac isn't really being used.
bub_find.get_bubbles(verbose=True, overlap_frac=0.5, multiprocess=True,
                     nsig=1.5, min_corr=0.7, min_overlap=0.8,
                     global_corr=0.5, min_channels=3, nprocesses=None)


# Make folder for the output
output_folder = os.path.join(datapath, "bubbles")
try:
    os.mkdir(output_folder)
except OSError:
    pass

# Save the bubble objects
bub_find.save_bubbles(folder=datapath, name=name)

# Create the catalog as an ecsv
catalog = bub_find.to_catalog()
catalog.write_table(os.path.join(output_folder,
                                 "{}_bubbles.ecsv".format(name)))

# Save the mask as an npy file. This isn't intended for normal output, but I
# want to be able to tweak parameters dependent on the expansion velocity
np.save(os.path.join(output_folder, "{}_bubble_mask.npy".format(name)),
        bub_find.mask)

# Save some plots of the distribution
# Bubble outlines only
fig = p.figure()
ax = fig.add_subplot(111)
ax = bub_find.visualize_bubbles(moment0=mom0, show=False, ax=ax,
                                plot_twoD_shapes=False)
fig.savefig(os.path.join(output_folder, "{}_mom0_bubbles.pdf".format(name)))
p.close()
# With the 2D regions
fig = p.figure()
ax = fig.add_subplot(111)
ax = bub_find.visualize_bubbles(moment0=mom0, show=False, ax=ax,
                                plot_twoD_shapes=True)
fig.savefig(os.path.join(output_folder,
                         "{}_mom0_bubbles_w_twoD.pdf".format(name)))
p.close()
