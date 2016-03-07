
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, fclusterdata, linkage

from log import overlap_metric


def cluster_2D_regions(twod_region_props, metric='position', cut_val=18):
    '''
    Cluster 2D Bubble regions by their postion or overlap.
    '''

    if metric is "postion":

        cluster_idx = fclusterdata(twod_region_props[1:3], cut_val,
                                   criterion='distance', method='complete')

        return cluster_idx

    elif metric is "overlap":

        if cut_val > 1.0:
            raise ValueError("cut_val <= 1 when metric is overlap.")

        sims = pdist(twod_region_props, metric=overlap_metric)
        sims[sims < 0] = 0.0
        link_mat = linkage(1 - sims, 'complete')
        cluster_idx = fcluster(link_mat, cut_val, criterion='distance')

        return cluster_idx
