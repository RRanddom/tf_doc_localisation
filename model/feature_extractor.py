from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

from model.mobilenet import mobilenet_v2

"""
TODO. support more backbone model
"""

slim = tf.contrib.slim

_MEAN_RGB = [123.15, 115.90, 103.06]

def _preprocess_subtract_imagenet_mean(inputs):
  """Subtract Imagenet mean RGB value."""
  mean_rgb = tf.reshape(_MEAN_RGB, [1, 1, 1, 3])
  return inputs - mean_rgb


def _preprocess_zero_mean_unit_range(inputs):
  """Map image values from [0, 255] to [-1, 1]."""
  return (2.0 / 255.0) * tf.to_float(inputs) - 1.0


def extract_features(images,
                     depth_multiplier=1.0,
                     final_endpoint=None,
                     num_classes=None,
                     weight_decay=0.00004,
                     is_training=False,
                     preprocess_images=True):
    """ Extracts features ~~~
    """
    images = tf.cast(images, tf.float32)
    if preprocess_images:
        images = _preprocess_subtract_imagenet_mean(images)

    with slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training, weight_decay=weight_decay)):
        features = mobilenet_v2.mobilenet(images, \
                                          num_classes=None, \
                                          final_endpoint=final_endpoint, \
                                          finegrain_classification_mode=True, \
                                          depth_multiplier=depth_multiplier)
                                          
    return features

    