
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, fclusterdata, linkage, dendrogram
from functools import partial

try:
    from sklearn.metrics import pairwise_distances
    from multiprocessing import cpu_count
    _sklearn_flag = True
except ImportError:
    Warning("sklearn must be installed to parallelize distance calculations.")
    _sklearn_flag = False

from log import overlap_metric
from utils import mode


def cluster_2D_regions(twod_region_props, metric='position', cut_val=18,
                       multiprocess=False, n_jobs=None, verbose=False):
    '''
    Cluster 2D Bubble regions by their postion or overlap.
    '''

    # Cluster on the x, y centre positions
    if metric is "position":

        cluster_idx = fclusterdata(twod_region_props[:, :2], cut_val,
                                   criterion='distance', method='complete')

    # Cluster on the spatial overlap of regions
    elif metric is "overlap":

        if cut_val > 1.0:
            raise ValueError("cut_val <= 1 when metric is overlap.")

        overlap_func = partial(overlap_metric, return_corr=True)

        if multiprocess and _sklearn_flag:
            if n_jobs is None:
                n_jobs = cpu_count()
            sims = pairwise_distances(twod_region_props, metric=overlap_func,
                                      n_jobs=n_jobs)
            # It's minorly not symmetric (~10^-10 differences), probably due
            # to pixelization. Force it to be symmetric.
            sym_sims = np.triu(sims) + np.triu(sims).T
            np.fill_diagonal(sym_sims, 0.0)
            # Convert to condensed form. linkage doesn't handle nxn dist
            # matrices properly
            sims = squareform(sym_sims)
        else:
            sims = pdist(twod_region_props, metric=overlap_func)
        sims[sims < 0] = 0.0

        link_mat = linkage(1 - sims, 'complete')

        if verbose:
            import matplotlib.pyplot as p
            dendrogram(link_mat)
            p.show()

        cluster_idx = fcluster(link_mat, cut_val, criterion='distance')

    # Cluster on the connectivity of regions in the spectral dimension
    elif metric is 'channel':

        Warning("cut_val forced to 1.0 for channel clustering.")

        cluster_idx = fclusterdata(twod_region_props[:, -1:], 1.0,
                                   criterion='distance', method='single',
                                   metric='cityblock')

    else:
        raise ValueError("metric must be 'position', 'channel', or 'overlap'.")

    return cluster_idx


def cluster_and_clean(twod_region_props, min_scatter=9, cut_val=None):
    '''
    Clean-up clusters of 2D regions. Real bubbles must be connected in
    velocity space. This function also looks for clusters that are closely
    related, and combines them.

    Parameters
    ----------
    twod_region_props : np.ndarray
        Array with the channel, y position, x position, semi-major radius,
        semi-minor radius, and position angle of the detected 2D blobs.
    min_scatter : float, optional
        Minimum distance to allow between the centres of regions in a cluster.

    Returns
    -------
    cluster_idx : np.ndarray
        The cluster ID for each of the given regions.
    '''

    # Create initial clusters based on the overlap correlation
    cluster_idx = cluster_2D_regions(twod_region_props,
                                     metric='overlap',
                                     cut_val=0.5, multiprocess=True)

    # Now we split clusters based on spectral connectivity.
    for clust in np.unique(cluster_idx[cluster_idx > 0]):

        posns = np.where(cluster_idx == clust)[0]

        if posns.size == 1:
            continue

        props = twod_region_props[posns]

        # Cluster on channel and split.
        spec_idx = cluster_2D_regions(props, metric='channel')

        # If not split is found, continue on
        if spec_idx.max() == 1:
            continue

        for idx in np.unique(spec_idx[spec_idx > 1]):
            cluster_idx[posns[spec_idx == idx]] = cluster_idx.max() + 1

    return cluster_idx


def cluster_brute_force(twod_region_props, cut_val=0.5, multiprocess=True,
                        n_jobs=None, min_multi_size=100):
    '''
    Do a brute force clustering of the regions
    '''

    cluster_idx = np.zeros_like(twod_region_props[:, 0])

    # Determine the channels which have regions defined in them
    chans = np.unique(twod_region_props[:, 5])

    overlap_func = partial(overlap_metric, return_corr=True)

    # Now loop through each channel's regions looking for significant overlap
    # with a region before it.
    for chan in chans[1:]:
        chan_regions_idx = np.where(twod_region_props[:, 5] == chan)[0]
        prev_regions_idx = np.where(twod_region_props[:, 5] == chan - 1)[0]

        all_overlaps = np.zeros((len(prev_regions_idx),
                                 len(chan_regions_idx)),
                                dtype=np.float)

        multi_conds = (multiprocess and _sklearn_flag and
                       all_overlaps.size >= min_multi_size)
        if multi_conds:
            if n_jobs is None:
                n_jobs = cpu_count()

            all_overlaps = \
                pairwise_distances(twod_region_props[prev_regions_idx],
                                   twod_region_props[chan_regions_idx],
                                   metric=overlap_func,
                                   n_jobs=n_jobs)
        else:
            for i, prev_idx in enumerate(prev_regions_idx):
                for j, idx in enumerate(chan_regions_idx):
                    all_overlaps[i, j] = \
                        overlap_func(twod_region_props[prev_idx],
                                     twod_region_props[idx])

        if not np.any(all_overlaps >= cut_val):
            continue

        for _ in range(len(prev_regions_idx)):

            i, j = np.unravel_index(all_overlaps.argmax(), all_overlaps.shape)

            if not all_overlaps[i, j] >= cut_val:
                break

            idx = prev_regions_idx[i]
            join_idx = chan_regions_idx[j]

            # Create a new cluster, or add to the existing one.
            if cluster_idx[idx] == 0:
                # Create a new cluster
                new_idx = cluster_idx.max() + 1
                cluster_idx[idx] = new_idx
                cluster_idx[join_idx] = new_idx
            else:
                cluster_idx[join_idx] = cluster_idx[idx]

            # Set that row and column to 0
            all_overlaps[i, :] = 0.0
            all_overlaps[:, j] = 0.0

    return cluster_idx
