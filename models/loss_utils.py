from typing import Optional, Tuple

import tensorflow as tf


def loss_mse(labels: tf.Tensor,
             preds: tf.Tensor,
             weights: Optional[tf.Tensor] = None,
             add_summaries: bool = True
             ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
    '''Computes the mean squared-error between preds and labels.

    Args
    - labels: tf.Tensor, shape [batch_size] or [batch_size, labels_dim]
    - preds: tf.Tensor, same shape as labels, type float32
    - weights: tf.Tensor, scalar or shape [batch_size], type float
    - add_sumaries: bool, whether to create summaries for the loss variables

    Returns
    - loss_total: tf.Tensor, scalar, loss_mse + loss_reg
    - loss_mse: tf.Tensor, scalar, mean squared-error loss over the batch
    - loss_reg: tf.Tensor, scalar, regularization loss
    - loss_summaries: tf.summary if add_summaries is True, None otherwise
    '''
    if weights is None:
        weights = 1.0
    loss_mse = tf.losses.mean_squared_error(
        labels=labels,
        predictions=preds,
        weights=weights,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    loss_total, loss_reg, loss_summaries = _loss_helper(
        loss_mse,
        loss_name='loss_mse',
        add_summaries=add_summaries)
    return loss_total, loss_mse, loss_reg, loss_summaries


def loss_xent(labels: tf.Tensor,
              logits: tf.Tensor,
              weights: Optional[tf.Tensor] = None,
              add_summaries: bool = True
              ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
    '''Takes a softmax over logits, then calculates cross-entropy loss with the
    given labels.

    Args
    - labels: tf.Tensor, shape [batch_size], type int32, elements in [0, num_classes)
    - logits: tf.Tensor, shape [batch_size, num_classes], type float32
    - weights: tf.Tensor, scalar or shape [batch_size], type float
    - add_sumaries: bool, whether to create summaries for the loss variables

    Returns
    - loss_total: tf.Tensor, scalar, loss_xent + loss_reg
    - loss_xent: tf.Tensor, scalar, mean cross-entropy loss over the batch
    - loss_reg: tf.Tensor, scalar, regularization loss
    - loss_summaries: tf.summary if add_summaries is True, None otherwise
    '''
    if weights is None:
        weights = 1.0
    loss_xent = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits,
        weights=weights,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    loss_total, loss_reg, loss_summaries = _loss_helper(
        loss_xent,
        loss_name='loss_cross_entropy',
        add_summaries=add_summaries)
    return loss_total, loss_xent, loss_reg, loss_summaries


def _loss_helper(loss: tf.Tensor,
                 loss_name: Optional[str] = None,
                 add_summaries: bool = False
                 ) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
    '''Helper function to get total loss and regularization loss, and add summaries for
    all losses.

    Args
    - loss: tf.Tensor, scalar, the loss without regularization
    - loss_name: str, name of loss for the summary, only needed if add_summaries=True
    - add_sumaries: bool, whether to create summaries for the loss variables

    Returns
    - loss_total: tf.Tensor, loss + loss_reg
    - loss_reg: tf.Tensor, the regularization loss
    - loss_summaries: tf.summary if add_summaries is True, None otherwise
    '''
    loss_total = tf.losses.get_total_loss(add_regularization_losses=True)
    loss_reg = tf.losses.get_regularization_loss()
    loss_summaries = None

    if add_summaries:
        loss_summaries = tf.summary.merge([
            tf.summary.scalar('loss_total', loss_total),
            tf.summary.scalar('loss_regularization_only', loss_reg),
            tf.summary.scalar(loss_name, loss)
        ])
    return loss_total, loss_reg, loss_summaries
