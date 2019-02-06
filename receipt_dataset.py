"""
Convert Image/Edgemap data to TFRecord file format with Example protos
"""

import math
import os.path
import re
import sys
import tensorflow as tf
import random
import build_data
from glob import glob

_NUM_SHARDS = 4

receipt_dir = "data/receipts"

midv_dir = "/Users/zjcneil_2018/Downloads/midv_500"
tfrecord_output_dir = "data/tfrecord"

receipt_eval_samples = ['receipt_459', 'receipt_485', 'receipt_374', 'receipt_400', 'receipt_395', 'receipt_540', 'receipt_558']
idcard_eval_samples = ['HA02', 'KA14', 'TS18', 'CA26', 'HA42', 'TS49', '*50_']


def _get_files(data, dataset_split):
    
    suffix = None
    if data == "image":
        suffix = ".jpg"
    else:
        suffix = ".txt"
        
    eval_files = []
    for eval_receipt in receipt_eval_samples:
        eval_files.extend(glob(os.path.join(receipt_dir, '*'+eval_receipt+'*'+suffix)))
    
    for eval_idcard in idcard_eval_samples:
        eval_files.extend(glob(os.path.join(midv_dir, '*'+eval_idcard+'*'+suffix)))
    
    filenames = None
    
    if dataset_split == "train":
        filenames = set(glob(os.path.join(receipt_dir, '*'+suffix)) + glob(os.path.join(midv_dir, '*'+suffix))) - set(eval_files)
    else:
        filenames = eval_files
    
    return sorted(filenames)

def _convert_dataset(dataset_split):
    """
    convert images and annots to tfrecord format.

    Args:
        dataset_split: "train" or "eval"
    """
    image_files = _get_files("image", dataset_split)
    txt_files = _get_files("txt", dataset_split)

    idx_list = list(range(len(image_files)))

    assert len(image_files) == len(txt_files), "number of image files and number of txts files must be same, if not, check your dataset dir"

    random.shuffle(idx_list)

    num_images = len(image_files)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))
    image_reader = build_data.ImageReader("jpg", channels=3)

    if not os.path.exists(tfrecord_output_dir):
        os.mkdir(tfrecord_output_dir)

    for shard_id in range(_NUM_SHARDS):
        shard_filename = "%s-%02d-of-%02d.tfrecord" % (
            dataset_split, shard_id, _NUM_SHARDS)
        output_filename = os.path.join(tfrecord_output_dir, shard_filename)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)

            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num_images, shard_id))
                sys.stdout.flush()

                image_data = tf.gfile.GFile(image_files[idx_list[i]], 'rb').read()
                height, width = image_reader.read_image_dims(image_data)

                with open(txt_files[idx_list[i]],'r') as txt_f:
                    content = txt_f.readline().strip().split(',')
                    points = [int(_) for _ in content]

                filename = os.path.basename(image_files[idx_list[i]])
                txtname = os.path.basename(txt_files[idx_list[i]])
                
                if filename.split('.')[0] != txtname.split('.')[0]:
                    raise ValueError('filename != txtname')

                example = build_data.image_label_to_tfexample(
                    image_data, filename, height, width, points)
                tfrecord_writer.write(example.SerializeToString())
        
        sys.stdout.write('\n')
        sys.stdout.flush()


def main(unused_argv):
  # Only support converting 'train' and 'eval' sets for now.
  for dataset_split in ['train', 'eval']:
      _convert_dataset(dataset_split)


def _parse_function(example_proto):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/height': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'image/width': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'image/points': tf.FixedLenFeature(
            (8,), tf.int64, default_value=None),
    }

    parsed_features = tf.parse_single_example(example_proto, keys_to_features)

    with tf.variable_scope('decoder'):
        input_image = tf.image.decode_image(parsed_features['image/encoded'], channels=3)
        input_height = parsed_features['image/height']
        input_width = parsed_features['image/width']
        points = parsed_features['image/points']
        image_name = parsed_features['image/filename']
    input_image.set_shape([None, None, 3])

    return input_image,image_name,input_height,input_width,points


def get_dataset_split(split_name):
    dataset_dir = tfrecord_output_dir
    file_pattern = 'train*.tfrecord' if split_name=='train' else 'eval*.tfrecord'
    filenames = tf.gfile.Glob(os.path.join(dataset_dir, file_pattern))
    dataset = tf.data.TFRecordDataset(filenames)
    return dataset.map(_parse_function)

if __name__ == '__main__':
    tf.app.run()
