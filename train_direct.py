r'''
This script trains ResNet CNN models to estimate wealth for DHS and LSMS
locations. Model checkpoints and TensorBoard training logs are saved to
`out_dir`.

Usage:
    python train_direct.py \
        --label_name wealthpooled \
        --model_name resnet --num_layers 18 \
        --lr_decay 0.96 --batch_size 64 \
        --gpu 0 --num_threads 5 \
        --cache train train_eval val \
        --augment True --eval_every 1 --print_every 40 \
        --ooc {ooc} --max_epochs {max_epochs} \
        --out_dir {out_dir} \
        --keep_frac {keep_frac} --seed {seed} \
        --experiment_name {experiment_name} \
        --dataset {dataset} \
        --ls_bands {ls_bands} --nl_band {nl_band} \
        --lr {lr} --fc_reg {reg} --conv_reg {reg} \
        --imagenet_weights_path {imagenet_weights_path} \
        --hs_weight_init {hs_weight_init}

Prerequisites: download TFRecords, process them, and create incountry folds. See
    `preprocessing/1_process_tfrecords.ipynb` and
    `preprocessing/2_create_incountry_folds.ipynb`.
'''
from __future__ import annotations

import argparse
import json
import os
from pprint import pprint
import time
from typing import Any, Optional

from batchers import batcher, tfrecord_paths_utils
from models.resnet_model import Hyperspectral_Resnet
from utils.run import get_full_experiment_name
from utils.trainer import RegressionTrainer

import numpy as np
import tensorflow as tf


ROOT_DIR = os.path.dirname(__file__)  # folder containing this file


def run_training(sess: tf.Session,
                 ooc: bool,
                 dataset: str,
                 keep_frac: float,
                 model_name: str,
                 model_params: dict[str, Any],
                 batch_size: int,
                 ls_bands: Optional[str],
                 nl_band: Optional[str],
                 label_name: str,
                 augment: bool,
                 learning_rate: float,
                 lr_decay: float,
                 max_epochs: int,
                 print_every: int,
                 eval_every: int,
                 num_threads: int,
                 cache: list[str],
                 out_dir: str,
                 init_ckpt_dir: Optional[str],
                 imagenet_weights_path: Optional[str],
                 hs_weight_init: Optional[str],
                 exclude_final_layer: bool
                 ) -> None:
    '''
    Args
    - sess: tf.Session
    - ooc: bool, whether to use out-of-country split
    - dataset: str
    - keep_frac: float
    - model_name: str, currently only 'resnet' is supported
    - model_params: dict
    - batch_size: int
    - ls_bands: one of [None, 'rgb', 'ms']
    - nl_band: one of [None, 'merge', 'split']
    - label_name: str, name of the label in the TFRecord file
    - augment: bool
    - learning_rate: float
    - lr_decay: float
    - max_epochs: int
    - print_every: int
    - eval_every: int
    - num_threads: int
    - cache: list of str, names of dataset splits to cache in RAM
    - out_dir: str, path to output directory for saving checkpoints and TensorBoard logs, must already exist
    - init_ckpt_dir: str, path to checkpoint dir from which to load existing weights
        - set to None to use ImageNet or random initialization
    - imagenet_weights_path: str, path to pre-trained weights from ImageNet
        - set to None to use saved ckpt or random initialization
    - hs_weight_init: str, one of [None, 'random', 'same', 'samescaled']
    - exclude_final_layer: bool, or None
    '''
    # ====================
    #    ERROR CHECKING
    # ====================
    assert os.path.exists(out_dir)

    if model_name == 'resnet':
        model_class = Hyperspectral_Resnet
    else:
        raise ValueError('Unknown model_name. Only "resnet" model currently supported.')

    # ====================
    #       BATCHERS
    # ====================
    if ooc:  # out-of-country split
        if 'dhs' in dataset.lower():
            train_tfrecord_paths = tfrecord_paths_utils.dhs_ooc(dataset, split='train')
            val_tfrecord_paths = tfrecord_paths_utils.dhs_ooc(dataset, split='val')
        else:
            raise ValueError('out-of-country w/ LSMS is not currently supported')

    else:  # in-country split
        if 'dhs' in dataset.lower():
            paths = tfrecord_paths_utils.dhs_incountry(dataset, splits=['train', 'val'])
        if 'lsms' in dataset.lower():
            paths = tfrecord_paths_utils.lsms_incountry(dataset, splits=['train', 'val'])

        train_tfrecord_paths = paths['train']
        val_tfrecord_paths = paths['val']

    num_train = len(train_tfrecord_paths)
    num_val = len(val_tfrecord_paths)

    # keep_frac affects sizes of both training and validation sets
    if keep_frac < 1.0:
        num_train = int(num_train * keep_frac)
        num_val = int(num_val * keep_frac)

        train_tfrecord_paths = np.random.choice(
            train_tfrecord_paths, size=num_train, replace=False)
        val_tfrecord_paths = np.random.choice(
            val_tfrecord_paths, size=num_val, replace=False)

    print('num_train:', num_train)
    print('num_val:', num_val)

    train_steps_per_epoch = int(np.ceil(num_train / batch_size))
    val_steps_per_epoch = int(np.ceil(num_val / batch_size))

    def get_batcher(tfrecord_paths: tf.Tensor, shuffle: bool, augment: bool,
                    epochs: int, cache: bool) -> batcher.Batcher:
        return batcher.Batcher(
            tfrecord_files=tfrecord_paths,
            label_name=label_name,
            ls_bands=ls_bands,
            nl_band=nl_band,
            batch_size=batch_size,
            epochs=epochs,
            normalize='DHS',  # TODO
            shuffle=shuffle,
            augment=augment,
            clipneg=True,
            cache=cache,
            num_threads=num_threads)

    train_tfrecord_paths_ph = tf.placeholder(tf.string, shape=[None])
    val_tfrecord_paths_ph = tf.placeholder(tf.string, shape=[None])

    with tf.name_scope('train_batcher'):
        train_batcher = get_batcher(
            train_tfrecord_paths_ph,
            shuffle=True,
            augment=augment,
            epochs=max_epochs,
            cache='train' in cache)
        train_init_iter, train_batch = train_batcher.get_batch()

    with tf.name_scope('train_eval_batcher'):
        train_eval_batcher = get_batcher(
            train_tfrecord_paths_ph,
            shuffle=False,
            augment=False,
            epochs=max_epochs + 1,  # may need extra epoch at the end of training
            cache='train_eval' in cache)
        train_eval_init_iter, train_eval_batch = train_eval_batcher.get_batch()

    with tf.name_scope('val_batcher'):
        val_batcher = get_batcher(
            val_tfrecord_paths_ph,
            shuffle=False,
            augment=False,
            epochs=max_epochs + 1,  # may need extra epoch at the end of training
            cache='val' in cache)
        val_init_iter, val_batch = val_batcher.get_batch()

    # ====================
    #        MODEL
    # ====================
    print('Building model...', flush=True)
    model_params['num_outputs'] = 1

    with tf.variable_scope(tf.get_variable_scope()) as model_scope:
        train_model = model_class(train_batch['images'], is_training=True, **model_params)
        train_preds = tf.reshape(train_model.outputs, shape=[-1], name='train_preds')

    with tf.variable_scope(model_scope, reuse=True):
        train_eval_model = model_class(train_eval_batch['images'], is_training=False, **model_params)
        train_eval_preds = tf.reshape(train_eval_model.outputs, shape=[-1], name='train_eval_preds')

    with tf.variable_scope(model_scope, reuse=True):
        val_model = model_class(val_batch['images'], is_training=False, **model_params)
        val_preds = tf.reshape(val_model.outputs, shape=[-1], name='val_preds')

    trainer = RegressionTrainer(
        train_batch, train_eval_batch, val_batch,
        train_model, train_eval_model, val_model,
        train_preds, train_eval_preds, val_preds,
        sess, train_steps_per_epoch, ls_bands, nl_band, learning_rate, lr_decay,
        out_dir, init_ckpt_dir, imagenet_weights_path,
        hs_weight_init, exclude_final_layer, image_summaries=False)

    # initialize the training dataset iterator
    sess.run([train_init_iter, train_eval_init_iter, val_init_iter], feed_dict={
        train_tfrecord_paths_ph: train_tfrecord_paths,
        val_tfrecord_paths_ph: val_tfrecord_paths
    })

    for epoch in range(max_epochs):
        if epoch % eval_every == 0:
            trainer.eval_train(max_nbatches=train_steps_per_epoch)
            trainer.eval_val(max_nbatches=val_steps_per_epoch)
        trainer.train_epoch(print_every)

    trainer.eval_train(max_nbatches=train_steps_per_epoch)
    trainer.eval_val(max_nbatches=val_steps_per_epoch)
    trainer.log_results()


def run_training_wrapper(**params: Any) -> None:
    '''
    params is a dict with keys matching the arguments from _parse_args()
    '''
    start = time.time()
    print('Current time:', start)

    # print all of the flags
    pprint(params)

    # parameters that might be 'None'
    none_params = ['ls_bands', 'nl_band', 'hs_weight_init',
                   'imagenet_weights_path', 'init_ckpt_dir']
    for p in none_params:
        if params[p] == 'None':
            params[p] = None

    # reset any existing graph
    tf.reset_default_graph()

    # set the random seeds
    seed = params['seed']
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # create the output directory if needed
    full_experiment_name = get_full_experiment_name(
        params['experiment_name'], params['batch_size'],
        params['fc_reg'], params['conv_reg'], params['lr'])
    out_dir = os.path.join(params['out_dir'], full_experiment_name)
    params_filepath = os.path.join(out_dir, 'params.json')
    if os.path.exists(params_filepath):
        print(f'Stopping. Found previous run at: {params_filepath}')
        return

    print(f'Outputs directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    with open(params_filepath, 'w') as config_file:
        json.dump(params, config_file, indent=4)

    # Create session
    # - MUST set os.environ['CUDA_VISIBLE_DEVICES'] before creating tf.Session
    if params['gpu'] is None:  # restrict to CPU only
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(params['gpu'])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model_params = {
        'fc_reg': params['fc_reg'],
        'conv_reg': params['conv_reg'],
        'use_dilated_conv_in_first_layer': False,
    }

    if params['model_name'] == 'resnet':
        model_params['num_layers'] = params['num_layers']

    run_training(
        sess=sess,
        ooc=params['ooc'],
        dataset=params['dataset'],
        keep_frac=params['keep_frac'],
        model_name=params['model_name'],
        model_params=model_params,
        batch_size=params['batch_size'],
        ls_bands=params['ls_bands'],
        nl_band=params['nl_band'],
        label_name=params['label_name'],
        augment=params['augment'],
        learning_rate=params['lr'],
        lr_decay=params['lr_decay'],
        max_epochs=params['max_epochs'],
        print_every=params['print_every'],
        eval_every=params['eval_every'],
        num_threads=params['num_threads'],
        cache=params['cache'],
        out_dir=out_dir,
        init_ckpt_dir=params['init_ckpt_dir'],
        imagenet_weights_path=params['imagenet_weights_path'],
        hs_weight_init=params['hs_weight_init'],
        exclude_final_layer=params['exclude_final_layer'])
    sess.close()

    end = time.time()
    print('End time:', end)
    print('Time elasped (sec.):', end - start)


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run end-to-end training.')

    # paths
    parser.add_argument(
        '--experiment_name', default='new_experiment',
        help='name of experiment being run')
    parser.add_argument(
        '--out_dir', default=os.path.join(ROOT_DIR, 'outputs/'),
        help='path to output directory for saving checkpoints and TensorBoard '
             'logs')

    # initialization
    parser.add_argument(
        '--init_ckpt_dir',
        help='path to checkpoint prefix from which to initialize weights')
    parser.add_argument(
        '--imagenet_weights_path',
        help='path to ImageNet weights for initialization')
    parser.add_argument(
        '--hs_weight_init', choices=[None, 'random', 'same', 'samescaled'],
        help='method for initializing weights of non-RGB bands in 1st conv '
             'layer')
    parser.add_argument(
        '--exclude_final_layer', action='store_true',
        help='whether to use checkpoint to initialize final layer')

    # learning parameters
    parser.add_argument(
        '--label_name', default='wealthpooled',
        help='name of label to use from the TFRecord files')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='batch size')
    parser.add_argument(
        '--augment', action='store_true',
        help='whether to use data augmentation')
    parser.add_argument(
        '--fc_reg', type=float, default=1e-3,
        help='Regularization penalty factor for fully connected layers')
    parser.add_argument(
        '--conv_reg', type=float, default=1e-3,
        help='Regularization penalty factor for convolution layers')
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate for optimizer')
    parser.add_argument(
        '--lr_decay', type=float, default=1.0,
        help='Decay rate of the learning rate')

    # high-level model control
    parser.add_argument(
        '--model_name', default='resnet', choices=['resnet'],
        help='name of model architecture')

    # resnet-only params
    parser.add_argument(
        '--num_layers', type=int, default=18, choices=[18, 34, 50],
        help='number of ResNet layers')

    # data params
    parser.add_argument(
        '--dataset', default='DHS_OOC_A',  # TODO: choices?
        help='dataset to use')
    parser.add_argument(
        '--ooc', action='store_true',
        help='whether to use out-of-country split')
    parser.add_argument(
        '--keep_frac', type=float, default=1.0,
        help='fraction of training data to use')
    parser.add_argument(
        '--ls_bands', choices=[None, 'rgb', 'ms'],
        help='Landsat bands to use')
    parser.add_argument(
        '--nl_band', choices=[None, 'merge', 'split'],
        help='nightlights band')

    # system
    parser.add_argument(
        '--gpu', type=int,
        help='which GPU to use')
    parser.add_argument(
        '--num_threads', type=int, default=1,
        help='number of threads for batcher')
    parser.add_argument(
        '--cache', nargs='*', default=[], choices=['train', 'train_eval', 'val'],
        help='list of datasets to cache in memory')

    # Misc
    parser.add_argument(
        '--max_epochs', type=int, default=150,
        help='maximum number of epochs for training')
    parser.add_argument(
        '--eval_every', type=int, default=1,
        help='evaluate the model on the validation set after every so many '
             'epochs of training')
    parser.add_argument(
        '--print_every', type=int, default=40,
        help='print training statistics after every so many steps')
    parser.add_argument(
        '--seed', type=int, default=123,
        help='seed for random initialization and shuffling')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_training_wrapper(**vars(args))
