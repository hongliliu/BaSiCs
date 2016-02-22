
import numpy as np

from basics.profile import profile_line


def azimuthal_profiles(image, centre, radius, ntheta=360, verbose=False):
    '''
    Calculate azimuthal profiles from the centre of a bubble to its edge.
    '''

    thetas = np.linspace(0.0, 2*np.pi, ntheta)

    profiles = []

    for i, theta in enumerate(thetas):

        end_pt = (centre[0] + 2*radius*np.sin(theta),
                  centre[1] + 2*radius*np.cos(theta))

        profile, dists = profile_line(image, centre, end_pt)

        profiles.append(profile)

        if verbose:
            import matplotlib.pyplot as p

            p.subplot(121)
            p.imshow(image, cmap='afmhot')
            p.plot(centre[1], centre[0], 'bD')
            p.plot(end_pt[1], end_pt[0], 'rD')
            p.xlim([0, image.shape[1]])
            p.ylim([0, image.shape[0]])

            p.subplot(122)
            p.plot(dists, profile, 'bD-')

            p.show()

    profiles = np.asarray(profiles)

    return profiles
