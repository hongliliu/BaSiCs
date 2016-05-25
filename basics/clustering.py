
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, fclusterdata, linkage, dendrogram
from functools import partial
from itertools import combinations

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


def cluster_brute_force(twod_region_props, min_corr=0.5, min_overlap=0.7,
                        global_corr=0.5,
                        multiprocess=True, n_jobs=None, min_multi_size=100):
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

        all_overlaps = np.zeros((2, len(prev_regions_idx),
                                 len(chan_regions_idx)),
                                dtype=np.float)

        multi_conds = (multiprocess and _sklearn_flag and
                       all_overlaps[0].size >= min_multi_size)
        if multi_conds:
            if n_jobs is None:
                n_jobs = cpu_count()

            all_overlaps[0] = \
                pairwise_distances(twod_region_props[prev_regions_idx],
                                   twod_region_props[chan_regions_idx],
                                   metric=overlap_func,
                                   n_jobs=n_jobs)
            all_overlaps[1] = \
                pairwise_distances(twod_region_props[prev_regions_idx],
                                   twod_region_props[chan_regions_idx],
                                   metric=overlap_metric,
                                   n_jobs=n_jobs)

        else:
            for i, prev_idx in enumerate(prev_regions_idx):
                for j, idx in enumerate(chan_regions_idx):
                    # Area correlation
                    all_overlaps[0, i, j] = \
                        overlap_func(twod_region_props[prev_idx],
                                     twod_region_props[idx])
                    # Area fractional overlap
                    all_overlaps[1, i, j] = \
                        overlap_metric(twod_region_props[prev_idx],
                                       twod_region_props[idx])

        any_corr = np.any(all_overlaps[0] >= min_corr)
        any_frac = np.any(all_overlaps[1] >= min_overlap)
        if not any_corr or not any_frac:
            continue

        # Blank all positions that don't satisfy the criteria
        good_mask = np.logical_and(all_overlaps[0] >= min_corr,
                                   all_overlaps[1] >= min_overlap)

        all_overlaps = all_overlaps * good_mask

        # The max number of matches is the smallest number of regions in the
        # two channels.
        # for _ in range(min(len(chan_regions_idx), len(prev_regions_idx))):
        while True:

            # They don't all have to match, so break when nothing is left.
            if not np.any(all_overlaps):
                break

            i, j = np.unravel_index(all_overlaps[0].argmax(),
                                    all_overlaps.shape[1:])

            idx = prev_regions_idx[i]
            join_idx = chan_regions_idx[j]

            # Create a new cluster, or add to the existing one.
            if cluster_idx[idx] == 0:
                # Create a new cluster
                new_idx = cluster_idx.max() + 1
                cluster_idx[idx] = new_idx
                cluster_idx[join_idx] = new_idx
            else:
                # Global connectivity check
                clust_overlaps = []
                for memb in np.where(cluster_idx == cluster_idx[idx])[0]:
                    # We already know this one is good.
                    if memb == idx:
                        continue
                    clust_overlaps.append(
                        overlap_func(twod_region_props[memb],
                                     twod_region_props[join_idx]))
                clust_overlaps = np.array(clust_overlaps)
                if np.any(clust_overlaps < global_corr):
                    all_overlaps[:, i, j] = 0.0
                    continue

                cluster_idx[join_idx] = cluster_idx[idx]

            # Set that row and column to 0
            all_overlaps[:, i, :] = 0.0
            all_overlaps[:, :, j] = 0.0

    return cluster_idx


def threeD_overlaps(bubbles, overlap_frac=0.8, overlap_corr=0.7,
                    min_chan_overlap=2,
                    multiprocess=True, join_overlap_frac=0.7,
                    join_overlap_corr=0.6, min_multi_size=100,
                    n_jobs=None):
    '''
    Overlap removal and joining of 3D bubbles.
    '''

    all_overlaps = np.zeros((len(bubbles), len(bubbles)),
                            dtype=np.float)
    remove_bubbles = []
    joined_bubbles = []

    metric = lambda one, two: one.overlap_with(two)

    # Calculate overlap between all pairs
    multi_conds = (multiprocess and _sklearn_flag and
                   all_overlaps[0].size >= min_multi_size)
    if multi_conds:
        if n_jobs is None:
            n_jobs = cpu_count()

        all_overlaps = \
            pairwise_distances(bubbles, metric=metric,
                               n_jobs=n_jobs)

    else:
        for i, j in combinations(range(len(bubbles)), 2):
            # Area fractional overlap
            all_overlaps[i, j] = metric(bubbles[i], bubbles[j])

    size_sort = np.argsort(np.array([bub.area for bub in bubbles]))[::-1]

    # Now loop through and start the rejection/join process
    for idx in size_sort:
        # Skip if it has satisfied a removal criterion
        if idx in remove_bubbles:
            continue

        overlaps = all_overlaps[idx]

        if not (overlaps >= overlap_frac).any():
            continue

        large_bubble = bubbles[idx]

        potential_removals = []

        # We need to make sure to only compare to smaller matches
        matches = np.where(overlaps >= overlap_frac)[0]
        smaller_idxs = []
        for match in matches:
            if large_bubble.area < bubbles[match].area:
                continue

            smaller_idxs.append(match)
        smaller_idxs = np.array(smaller_idxs)

        for small_idx in smaller_idxs:

            small_bubble = bubbles[small_idx]

            # Check spectral overlap
            start_overlap = \
                small_bubble.channel_start - large_bubble.channel_end
            end_overlap = \
                small_bubble.channel_end - large_bubble.channel_start
            # Checking for no spectral overlap, or complete
            # First two cases are for after and before the larger bubble.
            if start_overlap > 0 and end_overlap > 0:
                continue
            elif start_overlap < 0 and end_overlap < 0:
                continue
            elif start_overlap < 0 and end_overlap > 0:
                # Contained completely inside
                potential_removals.append(small_idx)
            else:
                corr_overlap = \
                    small_bubble.overlap_with(large_bubble,
                                              return_corr=True)

                # We now need to find the amount of channel overlap
                if start_overlap == 0 or end_overlap == 0:
                    # Join if overlapping enough
                    can_join = (overlaps[small_idx] >= join_overlap_frac) & \
                        (corr_overlap >= join_overlap_corr)
                    if can_join:
                        # Now we want to check if either of these bubbles has
                        # already been marked for joining
                        has_joined = False
                        if len(joined_bubbles) > 0:
                            for i in range(len(joined_bubbles)):
                                if idx in joined_bubbles[i]:
                                    joined_bubbles[i].append(small_idx)
                                    has_joined = True
                                    break
                                elif small_idx in joined_bubbles[i]:
                                    joined_bubbles[i].append(idx)
                                    has_joined = True
                                    break
                        if not has_joined:
                            joined_bubbles.append([idx, small_idx])
                    continue
                elif start_overlap < 0 and end_overlap > 0:
                    chan_overlap = -start_overlap
                elif start_overlap > 0 and end_overlap < 0:
                    chan_overlap = end_overlap
                elif -start_overlap == end_overlap:
                    chan_overlap = end_overlap
                else:
                    raise Warning("Check the bubble inputs. All potential"
                                  " cases should be handled, so this should "
                                  "not occur...")

                # If the channel overlap is enough, consider for removal
                if chan_overlap < min_chan_overlap:
                    # Not enough, don't touch either
                    continue
                elif small_bubble.channel_width > large_bubble.channel_width:
                    # If the correlation is high enough, remove the large one
                    if corr_overlap > overlap_corr:
                        remove_bubbles.append(idx)
                        continue
                    else:
                        potential_removals.append(small_idx)
                else:
                    larger_shell_frac = small_bubble.shell_fraction > \
                        large_bubble.shell_fraction
                    # Determine which should be removed
                    if corr_overlap > overlap_corr:
                        # If the smaller has the larger shell fraction, remove
                        # the larger
                        if larger_shell_frac:
                            remove_bubbles.append(idx)
                            continue

        # If the larger region was removed at all, don't add the potential
        # removals
        if idx in remove_bubbles:
            continue
        else:
            remove_bubbles.extend(potential_removals)

    # Make a list of the bubble objects to be joined
    bubbles_to_join = \
        [[bubbles[i] for i in ind] for ind in joined_bubbles]

    join_flattened = []
    for join in joined_bubbles:
        join_flattened.extend(join)

    # Remove all bubbles marked, and the joins, since they will be added back
    # on
    all_remove = list(set(remove_bubbles + join_flattened))
    bubbles = [bub for i, bub in enumerate(bubbles) if i not in all_remove]

    new_twoD_clusters = join_bubbles(bubbles_to_join)

    return bubbles, new_twoD_clusters


def join_bubbles(join_bubbles):
    '''
    Combine two bubble objects together.

    Parameters
    ----------
    join_bubbles : list of lists
        A list of the  bubbles to join.
    '''

    # Create a list of the twoD regions in each
    new_twoD_clusters = []

    for join in join_bubbles:
        new_cluster = []
        for bub in join:
            new_cluster.extend(bub.twoD_regions)
        new_twoD_clusters.append(new_cluster)

    return new_twoD_clusters
