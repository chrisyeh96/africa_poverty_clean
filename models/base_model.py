from abc import ABCMeta, abstractmethod
from typing import List, Optional

import tensorflow as tf


class BaseModel(metaclass=ABCMeta):
    '''The base class of models'''

    def __init__(self, inputs: tf.Tensor, num_outputs: Optional[int],
                 is_training: bool, fc_reg: float, conv_reg: float):
        '''
        Args
        - inputs: tf.Tensor, shape [batch_size, H, W, C], type float32
        - num_outputs: int, number of output classes
            set to None if we are only extracting features
        - is_training: bool, or tf.placeholder of type tf.bool
        - fc_reg: float, regularization for weights in the fully-connected layer
        - conv_reg: float, regularization for weights in the conv layers
        '''
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.is_training = is_training
        self.fc_reg = fc_reg
        self.conv_reg = conv_reg

        # in subclasses, these should be initialized during __init__()
        self.outputs: tf.Tensor = None  # tf.Tensor, shape [batch_size, num_outputs]
        self.features_layer: tf.Tensor = None  # tf.Tensor, shape [batch_size, num_features]

    @abstractmethod
    def init_from_numpy(self, path: str, sess: tf.Session, hs_weight_init: str) -> None:
        '''
        Args:
        - path: str, path to saved weights
        - sess: tf.Session
        - hs_weight_init: str
        '''
        raise NotImplementedError

    @abstractmethod
    def get_first_layer_weights(self) -> tf.Tensor:
        '''Gets the weights in the first layer of the CNN

        Returns: tf.Tensor
        '''
        raise NotImplementedError

    @abstractmethod
    def get_final_layer_weights(self) -> List[tf.Tensor]:
        '''Gets the weights in the final fully-connected layer after the conv layers.

        Returns: list of tf.Tensor
        '''
        raise NotImplementedError

    @abstractmethod
    def get_first_layer_summaries(self, ls_bands: Optional[str] = None,
                                  nl_band: Optional[str] = None) -> tf.Tensor:
        '''Creates the following summaries:
        - histogram of weights in 1st conv layer
        - (if model includes batch-norm layer) histogram of 1st batch-norm layer's moving mean

        Args
        - ls_bands: one of [None, 'rgb', 'ms'], if 'ms' then add separate histograms for RGB vs. other
            channel weights the first layer of the CNN
        - nl_band: one of [None, 'split', 'merge']

        Returns
        - summaries: tf.summary, merged summaries
        '''
        raise NotImplementedError
