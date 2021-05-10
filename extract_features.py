'''
This script runs trained CNN models to extract features from either the DHS or
LSMS satellite images.

Usage:
    python extract_features.py

Prerequisites:
1) download TFRecords, process them, and create incountry folds. See
    `preprocessing/1_process_tfrecords.ipynb` and
    `preprocessing/2_create_incountry_folds.ipynb`.
2) either train models (see README.md for instructions), or download model
    checkpoints into outputs/ directory
    TODO: elaborate here
'''
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
import json
import os
from typing import Optional

import numpy as np
import tensorflow as tf

from batchers import batcher, tfrecord_paths_utils
from models.resnet_model import Hyperspectral_Resnet
from utils.run import check_existing, run_extraction_on_models


OUTPUTS_ROOT_DIR = 'outputs'


# ====================
#      Parameters
# ====================
BATCH_SIZE = 128
KEEP_FRAC = 1.0
IS_TRAINING = False

DHS_MODELS: list[str] = [
    # put paths to DHS models here (relative to OUTPUTS_ROOT_DIR)
    # e.g., "dhs_ooc/DHS_OOC_A_..."
    'dhs_ooc/DHS_OOC_A_ms_samescaled_b64_fc01_conv01_lr0001',
    'dhs_ooc/DHS_OOC_B_ms_samescaled_b64_fc001_conv001_lr0001',
    'dhs_ooc/DHS_OOC_C_ms_samescaled_b64_fc001_conv001_lr001',
    'dhs_ooc/DHS_OOC_D_ms_samescaled_b64_fc001_conv001_lr01',
    'dhs_ooc/DHS_OOC_E_ms_samescaled_b64_fc01_conv01_lr001',

    'dhs_ooc/DHS_OOC_A_nl_random_b64_fc1.0_conv1.0_lr0001',
    'dhs_ooc/DHS_OOC_B_nl_random_b64_fc1.0_conv1.0_lr0001',
    'dhs_ooc/DHS_OOC_C_nl_random_b64_fc1.0_conv1.0_lr0001',
    'dhs_ooc/DHS_OOC_D_nl_random_b64_fc1.0_conv1.0_lr01',
    'dhs_ooc/DHS_OOC_E_nl_random_b64_fc1.0_conv1.0_lr0001',

    'dhs_ooc/DHS_OOC_A_rgb_same_b64_fc001_conv001_lr01',
    'dhs_ooc/DHS_OOC_B_rgb_same_b64_fc001_conv001_lr0001',
    'dhs_ooc/DHS_OOC_C_rgb_same_b64_fc001_conv001_lr0001',
    'dhs_ooc/DHS_OOC_D_rgb_same_b64_fc1.0_conv1.0_lr01',
    'dhs_ooc/DHS_OOC_E_rgb_same_b64_fc001_conv001_lr0001',

    # put paths to DHSNL models here (for transfer learning)
    # TODO
]

LSMS_MODELS: list[str] = [
    # put paths to LSMS models here
    # TODO
]


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

MODEL_PARAMS = {
    'fc_reg': 5e-3,  # this doesn't actually matter
    'conv_reg': 5e-3,  # this doesn't actually matter
    'num_layers': 18,
    'num_outputs': 1,
    'is_training': IS_TRAINING,
}

# ====================
# End Parameters
# ====================


def get_model_class(model_arch: str) -> Callable:
    if model_arch == 'resnet':
        model_class = Hyperspectral_Resnet
    else:
        raise ValueError('Unknown model_arch. Currently only "resnet" is supported.')
    return model_class


def get_batcher(dataset: str, ls_bands: str, nl_band: str, num_epochs: int,
                cache: bool) -> tuple[batcher.Batcher, int, dict]:
    '''Gets the batcher for a given dataset.

    Args
    - dataset: str, one of ['dhs', 'lsms'] # TODO
    - ls_bands: one of [None, 'ms', 'rgb']
    - nl_band: one of [None, 'merge', 'split']
    - num_epochs: int
    - cache: bool, whether to cache the dataset in memory if num_epochs > 1

    Returns
    - b: Batcher
    - size: int, length of dataset
    - feed_dict: dict, feed_dict for initializing the dataset iterator
    '''
    if dataset == 'dhs':
        tfrecord_paths = tfrecord_paths_utils.dhs()
    elif dataset == 'lsms':  # TODO
        tfrecord_paths = tfrecord_paths_utils.lsms()
    else:
        raise ValueError(f'dataset={dataset} is unsupported')

    size = len(tfrecord_paths)
    tfrecord_paths_ph = tf.placeholder(tf.string, shape=[size])
    feed_dict = {tfrecord_paths_ph: tfrecord_paths}

    if dataset == 'dhs':
        b = batcher.Batcher(
            tfrecord_files=tfrecord_paths_ph,
            label_name='wealthpooled',
            ls_bands=ls_bands,
            nl_band=nl_band,
            nl_label=None,
            batch_size=BATCH_SIZE,
            epochs=num_epochs,
            normalize='DHS',
            shuffle=False,
            augment=False,
            clipneg=True,
            cache=(num_epochs > 1) and cache,
            num_threads=5)
    else:  # LSMS, TODO
        raise NotImplementedError
        # b = delta_batcher.DeltaBatcher()

    return b, size, feed_dict


def read_params_json(model_dir: str, keys: Iterable[str]) -> tuple:
    '''Reads requested keys from json file at `model_dir/params.json`.

    Args
    - model_dir: str, path to model output directory containing params.json file
    - keys: list of str, keys to read from the json file

    Returns: tuple of values
    '''
    json_path = os.path.join(model_dir, 'params.json')
    with open(json_path, 'r') as f:
        params = json.load(f)
    result = tuple(params[k] for k in keys)
    return result


def main() -> None:
    for model_dirs in [DHS_MODELS, LSMS_MODELS]:
        if not check_existing(model_dirs,
                              outputs_root_dir=OUTPUTS_ROOT_DIR,
                              test_filename='features.npz'):
            print('Stopping')
            return

    # group models by batcher configuration and model_arch, where
    #   config = (dataset, ls_bands, nl_band, model_arch)
    all_models = {'dhs': DHS_MODELS, 'lsms': LSMS_MODELS}
    models_by_config: dict[
        tuple[str, Optional[str], Optional[str], str], list[str]
        ] = defaultdict(list)
    for dataset, model_dirs in all_models.items():
        for model_dir in model_dirs:
            ls_bands, nl_band, model_arch = read_params_json(
                model_dir=os.path.join(OUTPUTS_ROOT_DIR, model_dir),
                keys=['ls_bands', 'nl_band', 'model_name'])
            config = (dataset, ls_bands, nl_band, model_arch)
            models_by_config[config].append(model_dir)

    for config, model_dirs in models_by_config.items():
        dataset, ls_bands, nl_band, model_arch = config
        print('====== Current Config: ======')
        print('- dataset:', dataset)
        print('- ls_bands:', ls_bands)
        print('- nl_band:', nl_band)
        print('- model_arch:', model_arch)
        print('- number of models:', len(model_dirs))
        print()

        b, size, feed_dict = get_batcher(
            dataset=dataset, ls_bands=ls_bands, nl_band=nl_band,
            num_epochs=len(model_dirs), cache=True)
        batches_per_epoch = int(np.ceil(size / BATCH_SIZE))

        run_extraction_on_models(
            model_dirs,
            ModelClass=get_model_class(model_arch),
            model_params=MODEL_PARAMS,
            batcher=b,
            batches_per_epoch=batches_per_epoch,
            out_root_dir=OUTPUTS_ROOT_DIR,
            save_filename='features.npz',
            batch_keys=['labels', 'locs', 'years'],
            feed_dict=feed_dict)


if __name__ == '__main__':
    main()
