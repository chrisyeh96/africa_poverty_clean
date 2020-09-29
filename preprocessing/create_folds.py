from collections import defaultdict
import itertools
import os
import pickle
from pprint import pprint
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import sklearn.cluster

from utils.geo_plot import plot_locs  # ignore: E402


def create_folds(locs: np.ndarray,
                 min_dist: float,
                 fold_names: Iterable[str],
                 verbose: bool = True,
                 plot_largest_clusters: int = 0
                 ) -> Dict[str, np.ndarray]:
    '''Partitions locs into folds.

    Args
    - locs: np.array, shape [N, 2]
    - min_dist: float, minimum distance between folds
    - fold_names: list of str, names of folds
    - verbose: bool
    - plot_largest_clusters: int, number of largest clusters to plot

    Returns
    - folds: dict, fold name => sorted np.array of indices of locs belonging to that fold
    '''
    # there may be duplicate locs => we want to cluster based on unique locs
    unique_locs = np.unique(locs, axis=0)  # get unique rows

    # dict that maps each (lat, lon) tuple to a list of corresponding indices in
    # the locs array
    locs_to_indices = defaultdict(list)
    for i, loc in enumerate(locs):
        locs_to_indices[tuple(loc)].append(i)

    # any point within `min_dist` of another point belongs to the same cluster
    # - cluster_labels assigns a cluster index (0-indexed) to each loc
    # - a cluster label of -1 means that the point is an outlier
    _, cluster_labels = sklearn.cluster.dbscan(
        X=unique_locs, eps=min_dist, min_samples=2, metric='euclidean')

    # mapping: cluster number => list of indices of points in that cluster
    # - if cluster label is -1 (outlier), then treat that unique loc as its own cluster
    neg_counter = -1
    clusters_dict = defaultdict(list)
    for loc, c in zip(unique_locs, cluster_labels):
        indices = locs_to_indices[tuple(loc)]
        if c < 0:
            c = neg_counter
            neg_counter -= 1
        clusters_dict[c].extend(indices)

    # sort clusters by descending cluster size
    sorted_clusters = sorted(clusters_dict.keys(), key=lambda c: -len(clusters_dict[c]))

    # greedily assign clusters to folds
    folds: Dict[str, List[int]] = {f: [] for f in fold_names}
    for c in sorted_clusters:
        # assign points in cluster c to smallest fold
        f = min(folds, key=lambda f: len(folds[f]))
        folds[f].extend(clusters_dict[c])

    for f in folds:
        folds[f] = np.sort(folds[f])

    # plot the largest clusters
    for i in range(plot_largest_clusters):
        c = sorted_clusters[i]
        indices = clusters_dict[c]
        title = 'cluster {c}: {n} points'.format(c=c, n=len(indices))
        plot_locs(locs[indices], figsize=(4, 4), title=title)

    if verbose:
        _, unique_counts = np.unique(cluster_labels, return_counts=True)

        num_outliers = np.sum(cluster_labels == -1)
        outlier_offset = int(num_outliers > 0)
        max_cluster_size = np.max(unique_counts[outlier_offset:])  # exclude outliers

        print('num clusters:', np.max(cluster_labels) + 1)  # clusters are 0-indexed
        print('num outliers:', num_outliers)
        print('max cluster size (excl. outliers):', max_cluster_size)

        fig, ax = plt.subplots(1, 1, figsize=(5, 2.5), constrained_layout=True)
        ax.hist(unique_counts[outlier_offset:], bins=50)  # exclude outliers
        ax.set(xlabel='cluster size', ylabel='count')
        ax.set_yscale('log')
        ax.set_title('histogram of cluster sizes (excluding outliers)')
        ax.grid(True)
        plt.show()

    return folds


def verify_folds(folds: Dict[str, np.ndarray],
                 locs: np.ndarray,
                 min_dist: float,
                 max_index: Optional[int] = None
                 ) -> None:
    '''Verifies that folds do not overlap.

    Args
    - folds: dict, fold name => np.array of indices of locs belonging to that fold
    - locs: np.array, shape [N, 2], each row is [lat, lon]
    - min_dist: float, minimum distance between folds
    - max_index: int, all indices in range(max_index) should be included
    '''
    print('Size of each fold')
    pprint({f: len(indices) for f, indices in folds.items()})

    for fold, idxs in folds.items():
        assert np.all(np.diff(idxs) >= 0)  # check that indices are sorted

    # check that all indices are included
    if max_index is not None:
        assert np.array_equal(
            np.sort(np.concatenate(list(folds.values()))),
            np.arange(max_index))

    # check to ensure no overlap
    print('Minimum distance between each pair of folds')
    for a, b in itertools.combinations(folds.keys(), r=2):
        a_idxs = folds[a]
        b_idxs = folds[b]
        dists = scipy.spatial.distance.cdist(locs[a_idxs], locs[b_idxs], metric='euclidean')
        assert np.min(dists) > min_dist
        print(a, b, np.min(dists))


def create_split_folds(test_folds: Dict[str, np.ndarray],
                       fold_names: List[str],
                       ) -> Dict[str, Dict[str, np.ndarray]]:
    '''Creates a folds dict mapping each fold name (str) to another dict
    that maps each split (str) to a np.array of indices.

    folds = {
        'A': {
            'train': np.array([...]),
            'val': np.array([...]),
            'test': np.array([...])},
        ...
        'E': {...}
    }

    Args
    - test_folds: dict, fold name => sorted np.array of indices of locs belonging to that fold
    - fold_names: list of str, names of folds

    Returns
    - folds: dict, folds[f][s] is a np.array of indices for split s of fold f
    '''
    # create train/val/test splits
    folds: Dict[str, Dict[str, np.ndarray]] = {}
    for i, f in enumerate(fold_names):
        folds[f] = {}
        folds[f]['test'] = test_folds[f]

        val_f = fold_names[(i+1) % 5]
        folds[f]['val'] = test_folds[val_f]

        train_fs = [fold_names[(i+2) % 5], fold_names[(i+3) % 5], fold_names[(i+4) % 5]]
        folds[f]['train'] = np.sort(np.concatenate([test_folds[f] for f in train_fs]))

    return folds


def save_folds(folds_path: str,
               folds: Dict[str, Dict[str, np.ndarray]],
               check_exists: bool = True
               ) -> None:
    '''Saves folds dict to a pickle file at folds_path.

    Args
    - folds_path: str, path to pickle folds dict
    - folds: dict, folds[f][s] is a np.array of indices for split s of fold f
    - check_exists: bool, if True, verifies that existing pickle at folds_path
        matches the given folds
    '''
    if check_exists and os.path.exists(folds_path):
        with open(folds_path, 'rb') as p:
            existing_folds = pickle.load(p)
        assert set(existing_folds.keys()) == set(folds.keys())
        for f in existing_folds:
            for s in ['train', 'val', 'test']:
                assert np.array_equal(folds[f][s], existing_folds[f][s])
    else:
        with open(folds_path, 'wb') as p:
            pickle.dump(folds, p)
