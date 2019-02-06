import tensorflow as tf 
from utils.preprocess_utils import random_left_right_flip, random_up_down_flip, get_bbox_from_points

#def _preprocess_fn(input_image, image_name, input_height, input_width, points, is_training=False):

def preprocess_image_and_points(input_image, image_name, height, width, points, is_training=True, need_bbox=False):
    """Preprocesses the image and points.

    Args:
        input_image : image tensor. shape of [height, width, 3]
        image_name  : string tensor. name of the image.
        height: image height
        width : image width
        points: 4 points. ~
        is_training : If the preprocessing is used for training or not.
        need_bbox: should return bbox info.

    Returns:
        image_processed,
        points_processed,
        bbox

    Raises:
    """
    # input_image.set_shape([height, width, 3])
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)
    input_image = tf.cast(input_image, tf.float32)
    points = tf.reshape(points, [4,2])
    points = tf.cast(points, tf.float32)

    if is_training:
        input_image, points = random_left_right_flip(input_image, points, width, height)
        input_image, points = random_up_down_flip(input_image, points, width, height)

    if need_bbox:
        bbox = get_bbox_from_points(points)
        return input_image, points, bbox
    else:
        return input_image, points
