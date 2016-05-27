
from spectral_cube import SpectralCube
import os
import numpy as np
from basics.bubble_segment2D import BubbleFinder2D
from basics.utils import sig_clip

# As per Walter+08, there are issues with bkg subtracting the M51 cube
# However, the masking in BaSiCs can do a great job at finding the
# continuum sources (using otherwise empty channels), and masking them out

data_path = "/media/eric/Data_3/VLA/THINGS/NGC_3031/"

cube = SpectralCube.read(os.path.join(data_path, "NGC_3031_NA_CUBE_THINGS.FITS"))

# Find sigma in an empty channel
# The pt source and (mostly) the sidelobes cause sigma clipping to find a value
# that's about 1.5x too large. I'm going to assume 1 mJy/bm just to set the mask
# properties
sigma = 0.001 # sig_clip(cube[0].value, nsig=10) / 3.
bubble_1 = BubbleFinder2D(cube[0], channel=15, sigma=sigma, auto_cut=True)
# Set really high so it doesn't pick up edge artifacts
bubble_1.create_mask(region_min_nsig=20)

bubble_178 = BubbleFinder2D(cube[-1], channel=15, sigma=sigma, auto_cut=True)
bubble_178.create_mask(region_min_nsig=20)

# Now apply the mask, which will be broadcasted across all channels
masked_cube = \
    cube.with_mask(bubble_1.insert_in_shape(bubble_1.mask, cube.shape[1:]).astype(bool))
masked_cube = \
    masked_cube.with_mask(bubble_178.insert_in_shape(bubble_178.mask, cube.shape[1:]).astype(bool))

masked_cube.write("/media/eric/Data_3/VLA/THINGS/NGC_3031/NGC_3031_NA_CUBE_THINGS_PT_MASKED.FITS")
