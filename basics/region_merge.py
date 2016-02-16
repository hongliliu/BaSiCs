
import numpy as np
from itertools import izip
import matplotlib.pyplot as p


def region_merge(label_cube, img, overlap_frac=0.8):
    '''
    Merge regions at different wavelet scales. Different planes
    in the labeled cube correspond to the segmentation at different
    wavelet scales.
    '''

    final_mask = np.zeros_like(label_cube[0], dtype=int)

    i = 0
    for s_plane, l_plane in izip(label_cube[:-1], label_cube[1:]):
        s_labels = xrange(1, s_plane.max()+1)

        for s_lab in s_labels:

            # print(np.mean(img[s_plane == s_lab]))
            # print(np.std(img[s_plane == s_lab]))
            # p.subplot(121)
            # hist = p.hist(img[s_plane == s_lab])
            # p.subplot(122)
            # p.imshow(img, cmap='afmhot')
            # p.contour(s_plane == s_lab, colors='b')
            # raw_input("?")
            # p.clf()

            l_overlap_pts = l_plane[s_plane == s_lab]

            # Is there any overlap with the larger scale?
            if l_overlap_pts.max() == 0:
                final_mask[s_plane == s_lab] = final_mask.max() + 1
                continue

            # Calculate overlaps
            l_labs = np.unique(l_overlap_pts[l_overlap_pts.nonzero()])

            overlap = np.empty_like(l_labs, dtype=float)

            for i, l_lab in enumerate(l_labs):
                overlap[i] = float(np.sum(l_overlap_pts == l_lab)) /\
                     float(np.sum(s_plane == s_lab))

            # Plot those that will be absorbed
            # p.imshow(s_plane == s_lab)
            # for i in range(overlap.shape[0]):
            #     if overlap[i] >= overlap_frac:
            #         p.contour(l_plane == l_labs[i], colors='r')
            #     else:
            #         p.contour(l_plane == l_labs[i], colors='b')
            # raw_input("?")
            # p.clf()

            if not np.any(overlap >= overlap_frac):
                final_mask[s_plane == s_lab] = final_mask.max() + 1
                l_plane[s_plane == s_lab] = 0
                continue

            arg = np.argmax(overlap)

            new_pts = s_plane == s_lab

            l_plane[new_pts] = l_labs[arg]

        i += 1

    for lab in range(1, l_plane.max() + 1):
        final_mask[l_plane == lab] = final_mask.max() + 1

    return final_mask
