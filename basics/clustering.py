
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster, fclusterdata, linkage
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
                       multiprocess=False, n_jobs=None):
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
        else:
            sims = pdist(twod_region_props, metric=overlap_func)
        sims[sims < 0] = 0.0
        link_mat = linkage(1 - sims, 'complete')
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


def cluster_and_clean(twod_region_props, min_scatter=9):
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

    # Initial cluster is based on position of the centre
    cluster_idx = cluster_2D_regions(twod_region_props,
                                     metric='position',
                                     cut_val=twod_region_props[:, 2].max())

    # cluster_idx = cluster_2D_regions(twod_region_props,
    #                                  metric='overlap',
    #                                  cut_val=0.5, multiprocess=True)

    # Finally, we split based on position. At this point, there should be
    # a close cluster of regions and possibly some outliers with small radii
    # that give a complete overlap.
    for clust in np.unique(cluster_idx[cluster_idx > 0]):

        posns = np.where(cluster_idx == clust)[0]

        if posns.size == 1:
            continue

        props = twod_region_props[posns]

        # Determine max scatter in cluster from the mode of the major axes.
        maj_mode = max(mode(props[:, 2]), min_scatter)
        # maj_mode = min_scatter

        # Cluster on channel and split.
        pos_idx = cluster_2D_regions(props,
                                     metric='position', cut_val=maj_mode)

        # If not split is found, continue on
        if pos_idx.max() == 1:
            continue

        for idx in np.unique(pos_idx[pos_idx > 1]):
            cluster_idx[posns[pos_idx == idx]] = cluster_idx.max() + 1

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
