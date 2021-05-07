from __future__ import annotations

from collections.abc import Mapping
import time

import numpy as np
import tensorflow as tf


def analyze_tfrecord_batch(iter_init: tf.Operation,
                           batch_op: Mapping[str, tf.Tensor],
                           total_num_images: int,
                           nbands: int,
                           k: int
                           ) -> dict[str, np.ndarray]:
    '''Calculates per-band statistics.

    A good pixel is one where at least 1 band is > 0.

    Args
    - iter_init: tf.Op, operation to initialize iterator, or None if not needed
    - batch_op: dict, str -> tf.Tensor
        - 'images': tf.Tensor, type float32, shape [batch_size, 224, 224, nbands]
        - 'locs': tf.Tensor, type float32, shape [batch_size, 2], each row is [lat, lon]
        - 'years': tf.Tensor, type int32, shape [batch_size]
    - total_num_images: int
    - nbands: int

    Returns: dict
    - 'num_good_pixels': np.array, shape [total_num_images], type int, # of good pixels per image
    - 'mins': np.array, shape [nbands], type float64, min value per band
    - 'mins_nz': np.array, shape [nbands], type float64, min value per band excluding values <= 0
    - 'mins_goodpx': np.array, shape [nbands], type float64, min value per band among good pixels
    - 'maxs': np.array, shape [nbands], type float64, max value per band
    - 'sums': np.array, shape [nbands], type float64, sum of values per band, excluding values <= 0
    - 'sum_sqs': np.array, shape [nbands], type float64, sum of squared-values per band, excluding values <= 0
    - 'nz_pixels': np.array, shape [nbands], type int64, # of non-zero pixels per band
    '''
    images_count = 0

    # statistics for each band: min, max, sum, sum of squares, number of non-zero pixels
    mins = np.ones(nbands, dtype=np.float64) * np.inf
    mins_nz = np.ones(nbands, dtype=np.float64) * np.inf
    mins_goodpx = np.ones(nbands, dtype=np.float64) * np.inf
    maxs = np.zeros(nbands, dtype=np.float64)
    sums = np.zeros(nbands, dtype=np.float64)
    sum_sqs = np.zeros(nbands, dtype=np.float64)
    nz_pixels = np.zeros(nbands, dtype=np.int64)

    batch_times = []
    processing_times = []
    start = time.time()

    # number of `good pixels` in each image
    num_good_pixels: list[int] = []

    with tf.Session() as sess:
        if iter_init is not None:
            sess.run(iter_init)

        while True:
            try:
                batch_start_time = time.time()
                batch_np = sess.run(batch_op)
                img_batch, year_batch = \
                    batch_np['images'], batch_np['years']
                batch_size = len(img_batch)

                processing_start_time = time.time()
                batch_times.append(processing_start_time - batch_start_time)

                dmsp_mask = (year_batch < 2012)
                dmsp_bands = np.arange(nbands-1)
                viirs_mask = ~dmsp_mask
                viirs_bands = [i for i in range(nbands) if i != nbands-2]

                batch_goodpx = np.any(img_batch > 0, axis=3)
                num_good_pixels_per_image = np.sum(batch_goodpx, axis=(1,2))
                num_good_pixels.extend(num_good_pixels_per_image)

                img_batch_positive = np.where(img_batch <= 0, np.inf, img_batch)
                img_batch_nonneg = np.where(img_batch < 0, 0, img_batch)

                for mask, bands in [(dmsp_mask, dmsp_bands), (viirs_mask, viirs_bands)]:
                    if np.sum(mask) == 0: continue

                    imgs = img_batch[mask]
                    imgs_positive = img_batch_positive[mask]
                    imgs_nonneg = img_batch_nonneg[mask]

                    goodpx = batch_goodpx[mask]
                    imgs_goodpx = imgs[goodpx]  # shape [len(mask), nbands]

                    mins[bands] = np.minimum(mins[bands], np.min(imgs, axis=(0,1,2)))
                    mins_nz[bands] = np.minimum(mins_nz[bands], np.min(imgs_positive, axis=(0,1,2)))
                    mins_goodpx[bands] = np.minimum(mins_goodpx[bands], np.min(imgs_goodpx, axis=0))
                    maxs[bands] = np.maximum(maxs[bands], np.max(imgs, axis=(0,1,2)))

                    # use np.float64 to avoid significant loss of precision in np.sum
                    sums[bands] += np.sum(imgs_nonneg, axis=(0,1,2), dtype=np.float64)
                    sum_sqs[bands] += np.sum(imgs_nonneg ** 2, axis=(0,1,2), dtype=np.float64)
                    nz_pixels[bands] += np.sum(imgs > 0, axis=(0,1,2))

                processing_times.append(time.time() - processing_start_time)

                images_count += batch_size
                if images_count % 1024 == 0:
                    print(f'\rProcessed {images_count}/{total_num_images} images...', end='')
            except tf.errors.OutOfRangeError:
                break

    total_time = time.time() - start
    assert len(num_good_pixels) == images_count
    assert images_count == total_num_images

    print(f'\rFinished. Processed {images_count} images.')
    print('Time per batch - mean: {:0.3f}s, std: {:0.3f}s'.format(
        np.mean(batch_times), np.std(batch_times)))
    print('Time to process each batch - mean: {:0.3f}s, std: {:0.3f}s'.format(
        np.mean(processing_times), np.std(processing_times)))
    print('Total time: {:0.3f}s, Num batches: {}'.format(total_time, len(batch_times)))

    stats = {
        'num_good_pixels': np.array(num_good_pixels),
        'mins': mins,
        'mins_nz': mins_nz,
        'mins_goodpx': mins_goodpx,
        'maxs': maxs,
        'sums': sums,
        'sum_sqs': sum_sqs,
        'nz_pixels': nz_pixels
    }
    return stats


def per_band_mean_std(stats: Mapping[str, np.ndarray],
                      band_order: list[str]
                      ) -> tuple[dict[str, np.number], dict]:
    '''Calculates the per-band mean and standard deviation, only including
    "good pixels". A good pixel is one where at least 1 band is > 0.

    Args
    - stats: dict
        - 'num_good_pixels': np.array, shape [total_num_images], type int, # of good pixels per image
        - 'sums': np.array, shape [nbands], type float64, sum of values per band, excluding values <= 0
        - 'sum_sqs': np.array, shape [nbands], type float64, sum of squared-values per band, excluding values <= 0
    - band_order: list of str, names of bands
    '''
    num_good_pixels, sums, sum_sqs = [
        stats[k] for k in
        ['num_good_pixels', 'sums', 'sum_sqs']
    ]
    num_total_pixels = np.sum(num_good_pixels)
    means = sums / float(num_total_pixels)
    stds = np.sqrt(sum_sqs/float(num_total_pixels) - means**2)

    means = {
        band_name: means[b]
        for b, band_name in enumerate(band_order)
    }
    stds = {
        band_name: stds[b]
        for b, band_name in enumerate(band_order)
    }
    return means, stds


def print_analysis_results(stats: Mapping[str, np.ndarray],
                           band_order: list[str]) -> None:
    '''Prints per-band statistics based on different pixel criteria.

    Args
    - stats: dict, see the output of analyze_tfrecord_batch() above.
    - band_order: list of str, names of bands
    '''
    num_good_pixels, mins, mins_nz, mins_goodpx, maxs, sums, sum_sqs, nz_pixels = [
        stats[k] for k in
        ['num_good_pixels', 'mins', 'mins_nz', 'mins_goodpx', 'maxs', 'sums', 'sum_sqs', 'nz_pixels']
    ]
    images_count = len(num_good_pixels)
    total_pixels_per_band = images_count * (224 ** 2)  # per band

    print('Statistics including bad pixels')
    means = sums / float(total_pixels_per_band)
    stds = np.sqrt(sum_sqs/float(total_pixels_per_band) - means**2)
    for i, band_name in enumerate(band_order):
        print('Band {:8s} - mean: {:10.6f}, std: {:>9.6f}, min: {:>11.6g}, max: {:11.6f}'.format(
            band_name, means[i], stds[i], mins[i], maxs[i]))

    print('')
    print('Statistics ignoring any 0s and negative values')
    means = sums / nz_pixels
    stds = np.sqrt(sum_sqs/nz_pixels - means**2)
    avg_nz_pixels = nz_pixels.astype(np.float32) / images_count
    for i, band_name in enumerate(band_order):
        print('Band {:8s} - mean: {:10.6f}, std: {:>9.6f}, min: {:>11.6g}, max: {:11.6f}, mean_nz: {:0.6f}'.format(
            band_name, means[i], stds[i], mins_nz[i], maxs[i], avg_nz_pixels[i]))

    print('')
    print('Statistics excluding the bad pixels')
    num_total_pixels = np.sum(num_good_pixels)
    means = sums / float(num_total_pixels)
    stds = np.sqrt(sum_sqs/float(num_total_pixels) - means**2)
    for i, band_name in enumerate(band_order):
        print('Band {:8s} - mean: {:10.6f}, std: {:>9.6f}, min: {:>11.6g}, max: {:11.6f}'.format(
            band_name, means[i], stds[i], mins_goodpx[i], maxs[i]))
