from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional

import tensorflow as tf

from batchers.dataset_constants import MEANS_DICT, STD_DEVS_DICT


class Batcher():
    def __init__(self,
                 tfrecord_files: Iterable[str] | tf.Tensor,
                 label_name: Optional[str] = None,
                 scalar_features: Optional[Mapping[str, tf.DType]] = None,
                 ls_bands: Optional[str] = 'rgb',
                 nl_band: Optional[str] = None,
                 nl_label: Optional[str] = None,
                 batch_size: int = 1,
                 epochs: int = 1,
                 normalize: Optional[str] = None,
                 shuffle: bool = False,
                 augment: bool = False,
                 clipneg: bool = True,
                 cache: bool = False,
                 num_threads: int = 1):
        '''
        Args
        - tfrecord_files: list of str, or a tf.Tensor (e.g. tf.placeholder) of str
            - path(s) to TFRecord files containing satellite images
        - label_name: str, name of feature within TFRecords to use as label, or None
        - scalar_features: dict, maps names (str) of additional features within a TFRecord
            to their parsed types
        - ls_bands: one of [None, 'rgb', 'ms'], which Landsat bands to include in batch['images']
            - None: no Landsat bands
            - 'rgb': only the RGB bands
            - 'ms': all 7 Landsat bands
        - nl_band: one of [None, 'merge', 'split'], which NL bands to include in batch['images']
            - None: no nightlights band
            - 'merge': single nightlights band
            - 'split': separate bands for DMSP and VIIRS (if one is absent, then band is all 0)
        - nl_label: one of [None, 'center', 'mean']
            - None: do not include nightlights as a label
            - 'center': nightlight value of center pixel
            - 'mean': mean nightlights value
        - batch_size: int
        - epochs: int, number of epochs to repeat the dataset
        - shuffle: bool, whether to shuffle data, should be False when not training
        - augment: bool, whether to use data augmentation, should be False when not training
        - clipneg: bool, whether to clip negative values to 0
        - normalize: str, must be one of the keys of MEANS_DICT
            - if given, subtracts mean and divides by std-dev
        - cache: bool, whether to cache this dataset in memory
        - num_threads: int, number of threads to use for parallel processing
        '''
        self.tfrecord_files = tfrecord_files
        self.label_name = label_name
        self.scalar_features = scalar_features
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.augment = augment
        self.normalize = normalize
        self.clipneg = clipneg
        self.cache = cache
        self.num_threads = num_threads

        if ls_bands not in [None, 'rgb', 'ms']:
            raise ValueError(f'got {ls_bands} for "ls_bands"')
        self.ls_bands = ls_bands

        if normalize is not None and normalize not in MEANS_DICT:
            raise ValueError(f'got {normalize} for "normalize"')
        self.normalize = normalize

        if nl_band not in [None, 'merge', 'split']:
            raise ValueError(f'got {nl_band} for "nl_band"')
        self.nl_band = nl_band

        if nl_label not in [None, 'center', 'mean']:
            raise ValueError(f'got {nl_label} for "nl_label"')
        self.nl_label = nl_label

    def get_batch(self) -> tuple[tf.Operation, dict[str, tf.Tensor]]:
        '''Gets the tf.Tensors that represent a batch of data.

        Returns
        - iter_init: tf.Operation that should be run before first use
        - batch: dict, str -> tf.Tensor
            - 'images': tf.Tensor, shape [batch_size, H, W, C], type float32
                - C depends on the ls_bands and nl_band settings
            - 'locs': tf.Tensor, shape [batch_size, 2], type float32, each row is [lat, lon]
            - 'labels': tf.Tensor, shape [batch_size] or [batch_size, label_dim], type float32
                - shape [batch_size, 2] if self.label_name and self.nl_label are not None
            - 'years': tf.Tensor, shape [batch_size], type int32

        IMPLEMENTATION NOTE: The order of tf.data.Dataset.batch() and .repeat() matters!
            Suppose the size of the dataset is not evenly divisible by self.batch_size.
            If batch then repeat, i.e., `ds.batch(batch_size).repeat(num_epochs)`:
                the last batch of every epoch will be smaller than batch_size
            If repeat then batch, i.e., `ds.repeat(num_epochs).batch(batch_size)`:
                the boundaries between epochs are blurred, i.e., the dataset "wraps around"
        '''
        if self.shuffle:
            # shuffle the order of the input files, then interleave their individual records
            dataset = (
                tf.data.Dataset.from_tensor_slices(self.tfrecord_files)
                .shuffle(buffer_size=1000)
                .interleave(
                    lambda file_path: tf.data.TFRecordDataset(file_path, compression_type='GZIP'),
                    cycle_length=self.num_threads,
                    block_length=1,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE))
        else:
            # convert to individual records
            dataset = tf.data.TFRecordDataset(
                filenames=self.tfrecord_files,
                compression_type='GZIP',
                buffer_size=1024 * 1024 * 128,  # 128 MB buffer size
                num_parallel_reads=self.num_threads)

        # filter out unwanted TFRecords
        if getattr(self, 'filter_fn', None) is not None:
            dataset = dataset.filter(self.filter_fn)  # type: ignore

        # prefetch 2 batches at a time to smooth out the time taken to
        # load input files as we go through shuffling and processing
        dataset = dataset.prefetch(buffer_size=2 * self.batch_size)
        dataset = dataset.map(self.process_tfrecords, num_parallel_calls=self.num_threads)
        if self.nl_band == 'split':
            dataset = dataset.map(self.split_nl_band)

        if self.cache:
            dataset = dataset.cache()
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        if self.augment:
            dataset = dataset.map(self.augment_example)

        # batch then repeat => batches respect epoch boundaries
        # - i.e. last batch of each epoch might be smaller than batch_size
        dataset = dataset.batch(self.batch_size)
        if self.epochs > 1:
            dataset = dataset.repeat(self.epochs)

        # prefetch 2 batches at a time
        dataset = dataset.prefetch(2)

        iterator = dataset.make_initializable_iterator()
        batch = iterator.get_next()
        iter_init = iterator.initializer
        return iter_init, batch

    def process_tfrecords(self, example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
        '''
        Args
        - example_proto: a tf.train.Example protobuf

        Returns: dict {'images': img, 'labels': label, 'locs': loc, 'years': year, ...}
        - img: tf.Tensor, shape [224, 224, C], type float32
          - channel order is [B, G, R, SWIR1, SWIR2, TEMP1, NIR, NIGHTLIGHTS]
        - label: tf.Tensor, scalar or shape [2], type float32
          - not returned if both self.label_name and self.nl_label are None
          - [label, nl_label] (shape [2]) if self.label_name and self.nl_label are both not None
          - otherwise, is a scalar tf.Tensor containing the single label
        - loc: tf.Tensor, shape [2], type float32, order is [lat, lon]
        - year: tf.Tensor, scalar, type int32
          - default value of -1 if 'year' is not a key in the protobuf
        - may include other keys if self.scalar_features is not None
        '''
        img_bands = []  # bands that we want to include in the returned img
        if self.ls_bands == 'rgb':
            img_bands = ['BLUE', 'GREEN', 'RED']  # BGR order
        elif self.ls_bands == 'ms':
            img_bands = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR']
        ex_bands = img_bands.copy()  # bands that we want to parse from the tf.train.Example protobuf
        if (self.nl_band is not None) or (self.nl_label is not None):
            ex_bands += ['NIGHTLIGHTS']
            if self.nl_band is not None:
                img_bands += ['NIGHTLIGHTS']

        scalar_float_keys = ['lat', 'lon', 'year']
        if self.label_name is not None:
            scalar_float_keys.append(self.label_name)

        keys_to_features = {}
        for band in ex_bands:
            keys_to_features[band] = tf.io.FixedLenFeature(shape=[255**2], dtype=tf.float32)
        for key in scalar_float_keys:
            keys_to_features[key] = tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
        if self.scalar_features is not None:
            for key, dtype in self.scalar_features.items():
                keys_to_features[key] = tf.io.FixedLenFeature(shape=[], dtype=dtype)

        ex = tf.io.parse_single_example(example_proto, features=keys_to_features)
        loc = tf.stack([ex['lat'], ex['lon']])
        year = tf.cast(ex.get('year', -1), tf.int32)

        img = float('nan')
        if len(ex_bands) > 0:
            if self.normalize is not None:
                means = MEANS_DICT[self.normalize]
                std_devs = STD_DEVS_DICT[self.normalize]

            # for each band, reshape to (255, 255) and crop to (224, 224)
            # then subtract mean and divide by std dev
            for band in ex_bands:
                ex[band].set_shape([255 * 255])
                ex[band] = tf.reshape(ex[band], [255, 255])[15:-16, 15:-16]
                if self.clipneg:
                    ex[band] = tf.nn.relu(ex[band])
                if self.normalize:
                    if band == 'NIGHTLIGHTS':
                        ex[band] = tf.cond(
                            year < 2012,  # true = DMSP
                            true_fn=lambda: (ex[band] - means['DMSP']) / std_devs['DMSP'],
                            false_fn=lambda: (ex[band] - means['VIIRS']) / std_devs['VIIRS'])
                    else:
                        ex[band] = (ex[band] - means[band]) / std_devs[band]
            img = tf.stack([ex[band] for band in img_bands], axis=2)

        result = {'images': img, 'locs': loc, 'years': year}

        if self.label_name is not None:
            label = ex.get(self.label_name, float('nan'))
        if self.nl_label == 'mean':
            nl_label = tf.reduce_mean(ex['NIGHTLIGHTS'])
        elif self.nl_label == 'center':
            nl_label = ex['NIGHTLIGHTS'][112, 112]

        if self.label_name is None and self.nl_label is not None:
            result['labels'] = nl_label
        elif self.label_name is not None and self.nl_label is None:
            result['labels'] = label
        elif self.label_name is not None and self.nl_label is not None:
            result['labels'] = tf.stack([label, nl_label])

        if self.scalar_features is not None:
            for key in self.scalar_features:
                result[key] = ex[key]

        return result

    def split_nl_band(self, ex: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        '''Splits the NL band into separate DMSP and VIIRS bands.

        Args
        - ex: dict {'images': img, 'years': year, ...}
            - img: tf.Tensor, shape [H, W, C], type float32, final band is NL
            - year: tf.Tensor, scalar, type int32

        Returns: ex, with img updated to have 2 NL bands
        - img: tf.Tensor, shape [H, W, C], type float32, last two bands are [DMSP, VIIRS]
        '''
        assert self.nl_band == 'split'
        all_0 = tf.zeros(shape=[224, 224, 1], dtype=tf.float32, name='all_0')
        img = ex['images']
        year = ex['years']

        ex['images'] = tf.cond(
            year < 2012,
            # if DMSP, then add an all-0 VIIRS band to the end
            true_fn=lambda: tf.concat([img, all_0], axis=2),
            # if VIIRS, then insert an all-0 DMSP band before the last band
            false_fn=lambda: tf.concat([img[:, :, 0:-1], all_0, img[:, :, -1:]], axis=2)
        )
        return ex

    def augment_example(self, ex: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        '''Performs image augmentation (random flips + levels adjustments).
        Does not perform level adjustments on NL band(s).

        Args
        - ex: dict {'images': img, ...}
            - img: tf.Tensor, shape [H, W, C], type float32
                NL band depends on self.ls_bands and self.nl_band

        Returns: ex, with img replaced with an augmented image
        '''
        assert self.augment
        img = ex['images']

        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_flip_left_right(img)
        img = self.augment_levels(img)

        ex['images'] = img
        return ex

    def augment_levels(self, img: tf.Tensor) -> tf.Tensor:
        '''Perform random brightness / contrast on the image.
        Does not perform level adjustments on NL band(s).

        Args
        - img: tf.Tensor, shape [H, W, C], type float32
            - self.nl_band = 'merge' => final band is NL band
            - self.nl_band = 'split' => last 2 bands are NL bands

        Returns: tf.Tensor with data augmentation applied
        '''
        def rand_levels(image: tf.Tensor) -> tf.Tensor:
            # up to 0.5 std dev brightness change
            image = tf.image.random_brightness(image, max_delta=0.5)
            image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
            return image

        # only do random brightness / contrast on non-NL bands
        if self.ls_bands is not None:
            if self.nl_band is None:
                img = rand_levels(img)
            elif self.nl_band == 'merge':
                img_nonl = rand_levels(img[:, :, :-1])
                img = tf.concat([img_nonl, img[:, :, -1:]], axis=2)
            elif self.nl_band == 'split':
                img_nonl = rand_levels(img[:, :, :-2])
                img = tf.concat([img_nonl, img[:, :, -2:]], axis=2)
        return img


class UrbanBatcher(Batcher):
    def filter_fn(self, example_proto: tf.Tensor) -> tf.Tensor:
        '''
        Args
        - example_proto: a tf.train.Example protobuf

        Returns
        - predicate: tf.Tensor, type bool, True to keep, False to filter out
        '''
        keys_to_features = {
            'urban_rural': tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
        }
        ex = tf.io.parse_single_example(example_proto, features=keys_to_features)
        do_keep = tf.equal(ex['urban_rural'], 1.0)
        return do_keep


class RuralBatcher(Batcher):
    def filter_fn(self, example_proto: tf.Tensor) -> tf.Tensor:
        '''
        Args
        - example_proto: a tf.train.Example protobuf

        Returns
        - predicate: tf.Tensor, type bool, True to keep, False to filter out
        '''
        keys_to_features = {
            'urban_rural': tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
        }
        ex = tf.io.parse_single_example(example_proto, features=keys_to_features)
        do_keep = tf.equal(ex['urban_rural'], 0.0)
        return do_keep
