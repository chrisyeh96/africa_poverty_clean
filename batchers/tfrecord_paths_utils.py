from __future__ import annotations

from collections.abc import Iterable
from glob import glob
import os
import pickle
from typing import Optional

import numpy as np

from batchers.dataset_constants import SIZES, SURVEY_NAMES


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DHS_TFRECORDS_PATH_ROOT = os.path.join(ROOT_DIR, 'data/dhs_tfrecords')
DHSNL_TFRECORDS_PATH_ROOT = os.path.join(ROOT_DIR, 'data/dhsnl_tfrecords')
LSMS_TFRECORDS_PATH_ROOT = os.path.join(ROOT_DIR, 'data/lsms_tfrecords')


def dhs() -> np.ndarray:
    '''Gets a list of paths to all TFRecord files comprising the DHS dataset.

    Returns: np.array of str, sorted paths to TFRecord files
    '''
    return dhs_ooc(split='all', dataset='DHS_OOC_A')


def lsms() -> np.ndarray:
    '''Gets a list of paths to all TFRecord files comprising the LSMS dataset.

    Returns: np.array of str, sorted paths to TFRecord files
    '''
    raise NotImplementedError  # TODO


def dhsnl() -> np.ndarray:
    '''Gets a list of paths to all TFRecord files comprising the DHSNL dataset.

    Returns: np.array of str, sorted paths to TFRecord files
    '''
    glob_path = os.path.join(DHSNL_TFRECORDS_PATH_ROOT, '*', '*.tfrecord.gz')
    tfrecord_paths = np.sort(glob(glob_path))
    # assert len(tfrecord_paths) == SIZES['DHSNL']['all']  # TODO: uncomment this
    return tfrecord_paths


def dhs_ooc(dataset: str, split: str) -> np.ndarray:
    '''Gets a list of paths to TFRecords corresponding to the given split of
    a desired DHS dataset.

    Args
    - dataset: str, has format 'DHS_OOC_X' where 'X' is one of ['A', 'B', 'C', 'D', 'E']
    - splits: str, one of ['train', 'val', 'test', 'all']

    Returns: np.array of str, sorted paths to TFRecord files
    '''
    if split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [split]

    survey_names = SURVEY_NAMES[dataset]
    tfrecord_paths = []
    for s in splits:
        for country_year in survey_names[s]:
            glob_path = os.path.join(
                DHS_TFRECORDS_PATH_ROOT, country_year + '*', '*.tfrecord.gz')
            tfrecord_paths += glob(glob_path)
    assert len(tfrecord_paths) == SIZES[dataset][split]
    return np.sort(tfrecord_paths)


def lsms_ooc(cys: Optional[Iterable[str]] = None) -> list[str]:
    '''Gets a list of paths to TFRecords for a given list of LSMS surveys.

    Args
    - cys: list of 'country_year' str, order matters, or None for all

    Returns:
    - tfrecord_paths: list of str, paths to TFRecord files, order of country_years given by cys


    TODO: THIS FUNCTION NEEDS WORK!
    '''
    if cys is None:
        cys = sorted(SIZES['LSMS'].keys())
    tfrecord_paths = []
    for cy in cys:
        glob_path = os.path.join(LSMS_TFRECORDS_PATH_ROOT, cy, '*.tfrecord.gz')
        tfrecord_paths.extend(sorted(glob(glob_path)))
    expected_size = sum([SIZES['LSMS'][cy] for cy in cys])
    # assert len(tfrecord_paths) == expected_size  # TODO: uncomment this
    return tfrecord_paths


def _incountry(dataset: str, splits: Iterable[str], tfrecords_glob_path: str,
               folds_pickle_path: str
               ) -> dict[str, np.ndarray]:
    '''
    Args
    - dataset: str, format '*_incountry_X' where 'X' is one of
        ['A', 'B', 'C', 'D', 'E']
    - splits: list of str, from ['train', 'val', 'test', 'all']
    - tfrecords_glob_path: str, glob pattern for TFRecords
    - folds_pickle_path: str, path to pickle file containing incountry folds

    Returns
    - paths: dict, maps split (str) => sorted np.array of str paths

    Note: This is a little hacky, because it assumes that the list of TFRecord
        paths returned matches the order of the clusters used to create the
        folds pickle file. We know this to be true because both are
        sorted by the survey country_year, and the order of clusters within a
        survey are consistent. (The Google Earth Engine TFRecord export
        script preserves ordering within each survey.)
    '''
    all_tfrecord_paths = np.sort(glob(tfrecords_glob_path))
    assert len(all_tfrecord_paths) == SIZES[dataset]['all']

    fold = dataset[-1]
    with open(folds_pickle_path, 'rb') as f:
        incountry_folds = pickle.load(f)
        incountry_fold = incountry_folds[fold]

    paths: dict[str, np.ndarray] = {}
    for split in splits:
        if split == 'all':
            paths[split] == all_tfrecord_paths
        else:
            indices = incountry_fold[split]
            paths[split] = all_tfrecord_paths[indices]
        assert len(paths[split]) == SIZES[dataset][split]
    return paths


def dhs_incountry(dataset: str, splits: Iterable[str]) -> dict[str, np.ndarray]:
    '''
    Args
    - dataset: str, has format 'DHS_incountry_X' where 'X' is one of
        ['A', 'B', 'C', 'D', 'E']
    - splits: list of str, from ['train', 'val', 'test', 'all']

    Returns
    - paths: dict, maps split (str) => sorted np.array of str paths
    '''
    glob_path = os.path.join(DHS_TFRECORDS_PATH_ROOT, '*', '*.tfrecord.gz')
    folds_pickle_path = os.path.join(ROOT_DIR, 'data/dhs_incountry_folds.pkl')
    return _incountry(dataset=dataset, splits=splits,
                      tfrecords_glob_path=glob_path,
                      folds_pickle_path=folds_pickle_path)


def lsms_incountry(dataset: str, splits: Iterable[str]) -> dict[str, np.ndarray]:
    '''
    Args
    - dataset: str, has format 'LSMS_incountry_X' where 'X' is one of
        ['A', 'B', 'C', 'D', 'E']
    - splits: list of str, from ['train', 'val', 'test', 'all']

    Returns
    - paths: dict, maps split (str) => sorted np.array of str paths
    '''
    glob_path = os.path.join(LSMS_TFRECORDS_PATH_ROOT, '*', '*.tfrecord.gz')
    folds_pickle_path = os.path.join(ROOT_DIR, 'data/lsms_incountry_folds.pkl')
    return _incountry(dataset=dataset, splits=splits,
                      tfrecords_glob_path=glob_path,
                      folds_pickle_path=folds_pickle_path)


def lsms_pairs(indices_dict, delta_pairs_df, index_cols, other_cols=()):
    '''
    Args
    - indices_dict: dict, str => np.array of indices, the np.arrays are mutually exclusive
        or None to get all pairs
    - delta_pairs_df: pd.DataFrame
    - index_cols: list of str, [name of index1 column, name of index2 column]
    - other_cols: list of str, names of other columns to get

    Returns: np.array or dict
    - if indices_dict is None, returns: (paths, other1, ...)
        - paths: np.array, shape [N, 2], type str
        - others: np.array, shape [N], corresponds to columns from other_cols
    - otherwise, returns: (paths_dict, other_dict1, ...)
        - paths_dict: maps str => np.array, shape [X, 2], type str
            each row is [path1, path2], corresponds to TFRecords containing
            images of the same location such that year1 < year2
        - other_dicts: maps str => np.array, shape [X]
            corresponds to columns from other_cols

    TODO: THIS FUNCTION NEEDS WORK!
    '''
    assert len(index_cols) == 2
    tfrecord_paths = np.asarray(lsms_tfrecord_paths(SURVEY_NAMES['LSMS']))

    if indices_dict is None:
        ret = [None] * (len(other_cols) + 1)
        ret[0] = tfrecord_paths[delta_pairs_df[index_cols].values]
        for i, col in enumerate(other_cols):
            ret[i + 1] = delta_pairs_df[col].values
        return ret

    index1, index2 = index_cols
    return_dicts = [{} for i in range(len(other_cols) + 1)]
    paths_dict = return_dicts[0]

    for k, indices in indices_dict.items():
        mask = delta_pairs_df[index1].isin(indices)
        assert np.all(mask == delta_pairs_df[index2].isin(indices))
        paths_dict[k] = tfrecord_paths[delta_pairs_df.loc[mask, index_cols].values]
        for i, col in enumerate(other_cols):
            return_dicts[i + 1][k] = delta_pairs_df.loc[mask, col].values
    return return_dicts
