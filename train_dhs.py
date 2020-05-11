'''
This file trains ResNet-18 CNN models for estimating asset-wealth for DHS locations.
'''

import os
import pickle
from pprint import pprint
import time
from typing import Any, Dict, List, Optional

from batchers import batcher, dataset_constants
from models.resnet_model import Hyperspectral_Resnet
from utils.run import get_full_experiment_name
from utils.trainer import RegressionTrainer

import numpy as np
import tensorflow as tf


ROOT_DIR = os.path.dirname(__file__)  # folder containing this file


def run_training(sess: tf.Session,
                 ooc: bool,
                 batcher_type: str,
                 dataset: str,
                 keep_frac: float,
                 model_name: str,
                 model_params: Dict[str, Any],
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
                 cache: List[str],
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
    - batcher_type: str, type of batcher, one of ['base', 'urban', 'rural']
    - dataset: str, options depends on batcher_type
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
        train_tfrecord_paths = np.asarray(batcher.get_tfrecord_paths(dataset, 'train'))
        val_tfrecord_paths = np.asarray(batcher.get_tfrecord_paths(dataset, 'val'))

        sizes = {
            'base': dataset_constants.SIZES[dataset],
            'urban': dataset_constants.URBAN_SIZES[dataset],
            'rural': dataset_constants.RURAL_SIZES[dataset],
        }[batcher_type]

        assert len(train_tfrecord_paths) == sizes['train']
        assert len(val_tfrecord_paths) == sizes['val']

    else:  # in-country split
        if batcher_type != 'base':
            raise ValueError('incountry w/ non-base batcher is not supported')

        # This is a little hacky, because it assumes that the list of TFRecord
        # paths returned matches the order of the DHS clusters in
        # `data/dhs_clusters.csv`. We know this to be true because both are
        # sorted by the survey country_year, and the order of clusters within a
        # survey are consistent. (The Google Earth Engine TFRecord export
        # script preserves ordering within each survey.)
        all_tfrecord_paths = np.asarray(batcher.get_tfrecord_paths('DHS', 'all'))
        with open(os.path.join(ROOT_DIR, 'data/dhs_incountry_folds.pkl'), 'rb') as f:
            incountry_folds = pickle.load(f)
        assert len(all_tfrecord_paths) == dataset_constants.SIZES[dataset]['all']

        fold = dataset[-1]  # last letter of dataset
        train_indices = incountry_folds[fold]['train']
        val_indices = incountry_folds[fold]['val']

        train_tfrecord_paths = all_tfrecord_paths[train_indices]
        val_tfrecord_paths = all_tfrecord_paths[val_indices]

    num_train = len(train_tfrecord_paths)
    num_val = len(val_tfrecord_paths)

    # keep_frac affects sizes of both training and validation sets
    if keep_frac < 1.0:
        if batcher_type != 'base':
            raise ValueError('keep_frac < 1.0 w/ non-base batcher is not supported')

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
        BatcherClass = {
            'base': batcher.Batcher,
            'urban': batcher.UrbanBatcher,
            'rural': batcher.RuralBatcher,
        }[batcher_type]

        return BatcherClass(
            tfrecord_files=tfrecord_paths,
            label_name=label_name,
            ls_bands=ls_bands,
            nl_band=nl_band,
            batch_size=batch_size,
            epochs=epochs,
            normalize='DHS',
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
    params is a dict with keys matching the FLAGS defined below
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
    os.makedirs(out_dir, exist_ok=True)
    print(f'Outputs directory: {out_dir}')

    params_filepath = os.path.join(out_dir, 'params.txt')
    assert not os.path.exists(params_filepath), f'Stopping. Found previous run at: {params_filepath}'
    with open(params_filepath, 'w') as f:
        pprint(params, stream=f)
        pprint(f'Outputs directory: {out_dir}', stream=f)

    # Create session
    # - MUST set os.environ['CUDA_VISIBLE_DEVICES'] before creating tf.Session object
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
        batcher_type=params['batcher_type'],
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


def main(_: Any) -> None:
    params = {
        key: flags.FLAGS.__getattr__(key)
        for key in dir(flags.FLAGS)
    }
    run_training_wrapper(**params)


if __name__ == '__main__':
    flags = tf.app.flags

    # paths
    flags.DEFINE_string('experiment_name', 'new_experiment', 'name of the experiment being run')
    flags.DEFINE_string('out_dir', os.path.join(ROOT_DIR, 'outputs/'), 'path to output directory for saving checkpoints and TensorBoard logs')

    # initialization
    flags.DEFINE_string('init_ckpt_dir', None, 'path to checkpoint prefix from which to initialize weights (default None)')
    flags.DEFINE_string('imagenet_weights_path', None, 'path to ImageNet weights for initialization (default None)')
    flags.DEFINE_string('hs_weight_init', None, 'method for initializing weights of non-RGB bands in 1st conv layer, one of [None (default), "random", "same", "samescaled"]')
    flags.DEFINE_boolean('exclude_final_layer', False, 'whether to use checkpoint to initialize final layer')

    # learning parameters
    flags.DEFINE_string('label_name', 'wealthpooled', 'name of label to use from the TFRecord files')
    flags.DEFINE_integer('batch_size', 64, 'batch size')
    flags.DEFINE_boolean('augment', True, 'whether to use data augmentation')
    flags.DEFINE_float('fc_reg', 1e-3, 'Regularization penalty factor for fully connected layers')
    flags.DEFINE_float('conv_reg', 1e-3, 'Regularization penalty factor for convolution layers')
    flags.DEFINE_float('lr', 1e-3, 'Learning rate for optimizer')
    flags.DEFINE_float('lr_decay', 1.0, 'Decay rate of the learning rate (default 1.0 for no decay)')

    # high-level model control
    flags.DEFINE_string('model_name', 'resnet', 'name of the model to be used, currently only "resnet" is supported')

    # resnet-only params
    flags.DEFINE_integer('num_layers', 18, 'Number of ResNet layers, one of [18, 34, 50]')

    # data params
    flags.DEFINE_string('batcher_type', 'base', 'batcher, one of ["base", "urban", "rural"]')
    flags.DEFINE_string('dataset', 'DHS_A', 'dataset to use, options depend on batcher_type')  # TODO
    flags.DEFINE_boolean('ooc', True, 'whether to use out-of-country split')
    flags.DEFINE_float('keep_frac', 1.0, 'fraction of training data to use (default 1.0 uses all data)')
    flags.DEFINE_string('ls_bands', None, 'Landsat bands to use, one of [None (default), "rgb", "ms"]')
    flags.DEFINE_string('nl_band', None, 'nightlights band, one of [None (default), "merge", "split"]')

    # system
    flags.DEFINE_integer('gpu', None, 'which GPU to use (default None)')
    flags.DEFINE_integer('num_threads', 1, 'number of threads for batcher')
    flags.DEFINE_list('cache', [], 'comma-separated list (no spaces) of datasets to cache in memory, choose from [None, "train", "train_eval", "val"]')

    # Misc
    flags.DEFINE_integer('max_epochs', 150, 'maximum number of epochs for training')
    flags.DEFINE_integer('eval_every', 1, 'evaluate the model on the validation set after every so many epochs of training')
    flags.DEFINE_integer('print_every', 40, 'print training statistics after every so many steps')
    flags.DEFINE_integer('seed', 123, 'seed for random initialization and shuffling')

    tf.app.run()
