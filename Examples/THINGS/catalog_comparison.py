
import numpy as np
from astropy.table import Table
from astropy.modeling.models import Ellipse2D
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from spectral_cube import SpectralCube
from basics import Bubble2D, Bubble3D
import glob
import matplotlib.pyplot as p

'''
Compare the bubbles I find versus the Bagetakos catalog. Comparisons are based
primarily on spatial overlap of the bubbles and the central velocity they are
found at.

Other properties like the expansion velocity and bubble type will likely not
match well at this point.
'''


def get_closest_posn(posn, spatial_footprint):
    '''
    '''

    spatial_coords = SkyCoord(ra=spatial_footprint[1],
                              dec=spatial_footprint[0],
                              frame='fk5')

    min_posn = spatial_coords.separation(posn).argmin()

    twod_posn = np.unravel_index(min_posn, spatial_footprint[0].shape[::-1])

    return twod_posn


def get_maj_min(diam, ratio):
    '''
    Convert the diameter and ratio (min/maj) into the
    major/minor radii
    '''

    major = diam / (2 * np.sqrt(ratio))
    minor = (diam * np.sqrt(ratio)) / 2.

    return major, minor


def make_region(row, pixscale, cube):
    '''
    Create an ellipse from the Bagetakos catalogue.
    '''

    # These are in pc
    major, minor = get_maj_min(row["d"], row["Ratio"])

    # Convert to pixel lengths
    major /= pixscale.value
    minor /= pixscale.value

    # May need to shift PA by 90 deg
    pa = row["PA"]

    # Find pixel position using footprint from the cube
    center = SkyCoord(*row["Coords"].split(","), unit=u.deg)
    y, x = get_closest_posn(center, cube.spatial_coordinate_map)

    # Find the closest center channel
    cent_chan = cube.closest_spectral_channel(row["Vhel"] * u.km / u.s)

    props = np.array([y, x, major, minor, pa])

    # return Ellipse2D(True, x, y, major, minor, pa)
    return Bubble2D(props, channel=cent_chan)


def match_sources(set_one, set_two, return_corr=False):
    '''
    Look for spatial matches between 2 sets of bubbles
    '''

    dists = np.zeros((len(set_one), len(set_two)))

    for i, one in enumerate(set_one):
        for j, two in enumerate(set_two):
            dists[i, j] = one.overlap_with(two, return_corr=return_corr)

    return dists


if __name__ == "__main__":

    import os

    # Load in galaxy info
    execfile(os.path.expanduser("~/Dropbox/code_development/BaSiCs/Examples/THINGS/info_THINGS.py"))

    data_path = "/media/eric/Data_3/VLA/THINGS"

    # Load in Bagetakos catalog
    bagetakos_cat = \
        Table.read(os.path.join(data_path,
                                "bagetakos_bubbles_reformatted.ecsv"),
                   format='ascii.ecsv')

    region_overlaps = []

    # Compare the area correlation
    return_corr = True

    # What should constitute a match?
    min_corr = 0.3

    bubble_folder = "bubbles_NA"
    bubble_ro_folder = "bubbles_RO"

    overlaps = dict.fromkeys(galaxy_props.keys())
    overlaps_ro = dict.fromkeys(galaxy_props.keys())

    for i, key in enumerate(galaxy_props):

        # Skip 3031
        if key == "NGC_3031":
            continue

        props = galaxy_props[key]

        cube = SpectralCube.read(os.path.join(data_path, key,
                                              "{}_NA_CUBE_THINGS.FITS".format(key)))
        if cube._spectral_unit.is_equivalent(u.Hz):
            # Assuming RESTFREQ is defined in the header...
            cube = cube.with_spectral_unit(u.m / u.s, 'radio')

        pixscale = props["distance"].to(u.pc) * (np.pi / 180.) * \
            np.abs(cube.header["CDELT2"])

        mom0 = fits.getdata(os.path.join(data_path, key,
                                         "{}_NA_MOM0_THINGS.FITS".format(key)))

        regions = []
        for idx in np.where(bagetakos_cat["Name"] == props["name"])[0]:
            regions.append(make_region(bagetakos_cat[idx], pixscale, cube))

        # Now load in the pkl bubbles
        pkls = glob.glob(os.path.join(data_path, bubble_folder,
                                      key + "_bubbles_NA", "*.pkl"))
        bubbles = [Bubble3D.load_bubble(pk) for pk in pkls]

        pkls = glob.glob(os.path.join(data_path, bubble_ro_folder,
                                      key + "_bubbles_RO", "*.pkl"))
        bubbles_ro = [Bubble3D.load_bubble(pk) for pk in pkls]

        if len(bubbles) != 0:
            # continue
            dists = match_sources(regions, bubbles, return_corr=return_corr)

            overlaps[key] = \
                (dists.max(1) > min_corr).sum() / float(len(regions))
            print("NA Fraction with overlap")
            print(overlaps[key])
        else:
            print("NA Found no bubbles for: {}".format(props['name']))
            overlaps[key] = 0.0

        if len(bubbles_ro) != 0:
            # continue
            dists_ro = match_sources(regions, bubbles_ro,
                                     return_corr=return_corr)

            overlaps_ro[key] = \
                (dists_ro.max(1) > min_corr).sum() / float(len(regions))
            print("RO Fraction with overlap")
            print(overlaps_ro[key])
        else:
            print("RO Found no bubbles for: {}".format(props['name']))
            overlaps_ro[key] = 0.0

        ax = p.subplot(111)
        ax.imshow(mom0.squeeze(), origin='lower', cmap='afmhot')
        for region in regions:
            ax.add_patch(region.as_patch(fill=False, color='g', linewidth=2))
        for bub in bubbles:
            ax.add_patch(bub.as_patch(fill=False, color='b', linewidth=2))
        for bub in bubbles_ro:
            ax.add_patch(bub.as_patch(fill=False, color='grey', linewidth=2))
        p.draw()
        p.savefig(os.path.join(data_path, bubble_folder,
                               key + "_bubbles_comparison.pdf"))
        # raw_input(props['name'])
        p.clf()

        # break

p.plot(overlaps.values(), 'bD', label='NA')
p.plot(overlaps_ro.values(), 'go', label='RO')
p.xticks(np.arange(len(overlaps.keys())),
         overlaps.keys(), rotation='vertical')
p.ylabel("BaSiCs Fraction Overlap with Bagetakos")
p.legend()
p.savefig(os.path.join(data_path, bubble_folder,
                       "fraction_matching_bubbles.pdf"))
# p.show()
p.clf()
