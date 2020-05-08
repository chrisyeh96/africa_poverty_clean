'''
This file trains ResNet-18 CNN models for estimating nightlights given
multi-spectral daytime satellite imagery.
'''

from glob import glob
import os
from pprint import pprint
from typing import Any, Dict, List, Optional

from batchers import batcher, dataset_constants
from models.base_model import BaseModel
from models.resnet_model import Hyperspectral_Resnet
from utils.run import get_full_experiment_name, make_log_and_ckpt_dirs
from utils.trainer import RegressionTrainer

import numpy as np
import tensorflow as tf


ROOT_DIR = os.path.dirname(__file__)  # folder containing this file


def run_training(sess: tf.Session,
                 dataset: str,
                 model_name: str,
                 model_params: Dict[str, Any],
                 batch_size: int,
                 ls_bands: Optional[str],
                 nl_label: str,
                 augment: bool,
                 learning_rate: float,
                 lr_decay: float,
                 max_epochs: int,
                 print_every: int,
                 eval_every: int,
                 num_threads: int,
                 cache: List[str],
                 log_dir: str,
                 save_ckpt_dir: str,
                 init_ckpt_dir: Optional[str],
                 imagenet_weights_path: Optional[str],
                 hs_weight_init: Optional[str],
                 exclude_final_layer: bool
                 ) -> None:
    '''
    Args
    - sess: tf.Session
    - dataset: str
    - model_name: str
    - model_params: dict
    - batch_size: int
    - ls_bands: one of [None, 'rgb', 'ms']
    - nl_label: str, one of ['center', 'mean']
    - augment: bool
    - learning_rate: float
    - lr_decay: float
    - max_epochs: int
    - print_every: int
    - eval_every: int
    - num_threads: int
    - cache: list of str
    - log_dir: str, path to directory to save logs for TensorBoard, must already exist
    - save_ckpt_dir: str, path to checkpoint dir for saving weights
        - intermediate dirs must already exist
    - init_ckpt_dir: str, path to checkpoint dir from which to load existing weights
        - set to empty string '' to use ImageNet or random initialization
    - imagenet_weights_path: str, path to pre-trained weights from ImageNet
        - set to empty string '' to use saved ckpt or random initialization
    - hs_weight_init: str, one of [None, 'random', 'same', 'samescaled']
    - exclude_final_layer: bool, or None
    '''
    ####################################
    #          ERROR CHECKING          #
    ####################################
    assert os.path.exists(log_dir)
    assert os.path.exists(os.path.dirname(save_ckpt_dir))

    if model_name == 'resnet':
        model_class = Hyperspectral_Resnet
    else:
        raise ValueError('Unknown model_name. Only "resnet" model currently supported.')

    ##############################
    #          BATCHERS          #
    ##############################
    tfrecord_files_glob = os.path.join(batcher.DHSNL_TFRECORDS_PATH_ROOT, '*', '*.tfrecord.gz')
    all_tfrecord_files = np.sort(glob(tfrecord_files_glob))
    assert len(all_tfrecord_files) == dataset_constants.SIZES[dataset]['all']

    def get_batcher(indices: np.ndarray, shuffle: bool, augment: bool,
                    epochs: int, cache: bool) -> batcher.Batcher:
        return batcher.Batcher(
            tfrecord_files=all_tfrecord_files[indices],
            ls_bands=ls_bands,
            nl_label=nl_label,
            batch_size=batch_size,
            epochs=epochs,
            normalize=dataset,
            shuffle=shuffle,
            augment=augment,
            clipneg=True,
            cache=cache,
            num_threads=num_threads)

    all_indices = np.random.permutation(len(all_tfrecord_files))
    num_train = int(len(all_tfrecord_files) * 0.9)
    steps_per_epoch = num_train * 1.0 / batch_size
    train_indices = all_indices[:num_train]
    val_indices = all_indices[num_train:]

    with tf.name_scope('train_batcher'):
        train_batcher = get_batcher(train_indices, shuffle=True, augment=augment,
                                    epochs=max_epochs, cache=False)
        train_init_iter, train_batch = train_batcher.get_batch()

    with tf.name_scope('train_eval_batcher'):
        # shuffle, because we are sampling from this train_eval batcher to get estimates of
        # our training mse / r^2 values, instead of evaluating over all of the training set
        train_eval_batcher = get_batcher(train_indices, shuffle=True, augment=False,
                                         epochs=max_epochs, cache=False)
        train_eval_init_iter, train_eval_batch = train_eval_batcher.get_batch()

    with tf.name_scope('val_batcher'):
        val_batcher = get_batcher(val_indices, shuffle=False, augment=False,
                                  epochs=1, cache=True)
        val_init_iter, val_batch = val_batcher.get_batch()

    ###########################
    #          MODEL          #
    ###########################
    print('Building model...', flush=True)
    model_params['num_outputs'] = 2  # model predicts both DMSP and VIIRS values

    with tf.variable_scope(tf.get_variable_scope()) as model_scope:
        train_model = model_class(train_batch['images'], is_training=True, **model_params)

    with tf.variable_scope(model_scope, reuse=True):
        train_eval_model = model_class(train_eval_batch['images'], is_training=False, **model_params)

    with tf.variable_scope(model_scope, reuse=True):
        val_model = model_class(val_batch['images'], is_training=False, **model_params)

    def get_nl_preds(model: BaseModel, years: tf.Tensor) -> tf.Tensor:
        return tf.where(years < 2012, model.outputs[:, 0], model.outputs[:, 1])

    train_preds = get_nl_preds(train_model, train_batch['years'])
    train_eval_preds = get_nl_preds(train_eval_model, train_eval_batch['years'])
    val_preds = get_nl_preds(val_model, val_batch['years'])

    trainer = RegressionTrainer(
        train_batch, train_eval_batch, val_batch,
        train_model, train_eval_model, val_model,
        train_preds, train_eval_preds, val_preds,
        sess, train_steps_per_epoch, ls_bands, None, learning_rate, lr_decay,
        log_dir, save_ckpt_dir, init_ckpt_dir, imagenet_weights_path,
        hs_weight_init, None, image_summaries=False)

    # initialize the training dataset iterators
    sess.run([train_init_iter, train_eval_init_iter])

    for epoch in range(max_epochs):
        if epoch % eval_every == 0:
            trainer.eval_train(max_nbatches=200)
            trainer.eval_val(max_nbatches=val_steps_per_epoch)
        trainer.train_epoch(print_every)

    trainer.eval_train(max_nbatches=500)
    trainer.eval_val(max_nbatches=val_steps_per_epoch)

    csv_log_path = os.path.join(log_dir, 'results.csv')
    trainer.log_results(csv_log_path)


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

    # create the log and checkpoint directories if needed
    full_experiment_name = get_full_experiment_name(
        params['experiment_name'], params['batch_size'],
        params['fc_reg'], params['conv_reg'], params['lr'])
    log_dir, ckpt_prefix = make_log_and_ckpt_dirs(
        params['log_dir'], params['ckpt_dir'], full_experiment_name)
    print(f'Checkpoint prefix: {ckpt_prefix}')

    params_filepath = os.path.join(log_dir, 'params.txt')
    assert not os.path.exists(params_filepath), f'Stopping. Found previous run at: {params_filepath}'
    with open(params_filepath, 'w') as f:
        pprint(params, stream=f)
        pprint(f'Checkpoint prefix: {ckpt_prefix}', stream=f)

    # Create session
    # - MUST set up os.environ['CUDA_VISIBLE_DEVICES'] before creating the tf.Session object
    if params['gpu_usage'] == 0: # restrict to CPU only
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
        dataset=params['dataset'],
        model_name=params['model_name'],
        model_params=model_params,
        batch_size=params['batch_size'],
        ls_bands=params['ls_bands'],
        nl_label=params['nl_label'],
        augment=params['augment'],
        learning_rate=params['lr'],
        lr_decay=params['lr_decay'],
        max_epochs=params['max_epochs'],
        print_every=params['print_every'],
        eval_every=params['eval_every'],
        num_threads=params['num_threads'],
        log_dir=log_dir,
        save_ckpt_dir=ckpt_prefix,
        init_ckpt_dir=params['init_ckpt_dir'],
        imagenet_weights_path=params['imagenet_weights_path'],
        hs_weight_init=params['hs_weight_init'])
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
    flags.DEFINE_string('ckpt_dir', os.path.join(ROOT_DIR, 'ckpts/'), 'checkpoint directory')
    flags.DEFINE_string('log_dir', os.path.join(ROOT_DIR, 'logs/'), 'log directory')

    # initialization
    flags.DEFINE_string('init_ckpt_dir', None, 'path to checkpoint prefix from which to initialize weights (default None)')
    flags.DEFINE_string('imagenet_weights_path', None, 'path to ImageNet weights for initialization (default None)')
    flags.DEFINE_string('hs_weight_init', None, 'method for initializing weights of non-RGB bands in 1st conv layer, one of [None (default), "random", "same", "samescaled"]')

    # learning parameters
    flags.DEFINE_integer('batch_size', 64, 'batch size')
    flags.DEFINE_boolean('augment', True, 'whether to use data augmentation')
    flags.DEFINE_float('fc_reg', 1e-3, 'Regularization penalty factor for fully connected layers')
    flags.DEFINE_float('conv_reg', 1e-3, 'Regularization penalty factor for convolution layers')
    flags.DEFINE_float('lr', 1e-3, 'Learning rate for optimizer')
    flags.DEFINE_float('lr_decay', 1.0, 'Decay rate of the learning rate (default 1.0 for no decay)')

    # high-level model control
    flags.DEFINE_string('model_name', 'resnet', 'name of the model to be used, (default "resnet")')

    # resnet-only params
    flags.DEFINE_integer('num_layers', 18, 'Number of ResNet layers, one of [18 (default), 34, 50]')

    # data params
    flags.DEFINE_string('dataset', 'DHS_NL', 'dataset to use, options depend on batcher (default "DHS_NL")')
    flags.DEFINE_string('ls_bands', None, 'Landsat bands to use, one of [None (default), "rgb", "ms"]')
    flags.DEFINE_string('nl_label', 'center', 'what nightlight value to train on, one of ["center" (default), "mean"]')

    # system
    flags.DEFINE_integer('gpu', None, 'which GPU to use (default None)')
    flags.DEFINE_integer('num_threads', 1, 'number of threads for batcher (default 1)')
    flags.DEFINE_list('cache', [], 'comma-separated list (no spaces) of datasets to cache in memory, choose from [None, "train", "train_eval", "val"]')

    # Misc
    flags.DEFINE_integer('max_epochs', 150, 'maximum number of epochs for training (default 150)')
    flags.DEFINE_integer('eval_every', 1, 'evaluate the model on the validation set after every so many epochs of training (default 1)')
    flags.DEFINE_integer('print_every', 40, 'print training statistics after every so many steps (default 40)')
    flags.DEFINE_integer('seed', 123, 'seed for random initialization and shuffling (default 123)')

    tf.app.run()
