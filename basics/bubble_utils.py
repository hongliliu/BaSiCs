
import numpy as np

from basics.profile import profile_line


def azimuthal_profiles(image, blob, ntheta=360, verbose=False,
                       extend_factor=2):
    '''
    Calculate azimuthal profiles from the centre of a bubble to its edge.
    '''

    y0, x0, a, b, pa = blob.copy()

    a *= extend_factor
    b *= extend_factor

    thetas = np.linspace(0.0, 2*np.pi, ntheta)

    profiles = []

    for i, theta in enumerate(thetas):

        end_pt = (y0 + a*np.cos(theta)*np.sin(pa) - b*np.sin(theta)*np.cos(pa),
                  x0 + a*np.cos(theta)*np.cos(pa) - b*np.sin(theta)*np.sin(pa))

        profile, dists = profile_line(image, (y0, x0), end_pt)

        profiles.append(profile)

        if verbose:
            import matplotlib.pyplot as p

            p.subplot(121)
            p.imshow(image, cmap='afmhot')
            p.plot(x0, y0, 'bD')
            p.plot(end_pt[1], end_pt[0], 'rD')
            p.xlim([0, image.shape[1]])
            p.ylim([0, image.shape[0]])

            p.subplot(122)
            p.plot(dists, profile, 'bD-')

            p.show()

    profiles = np.asarray(profiles)

    return profiles
