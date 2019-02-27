import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from glob import glob
from model.keypoints_heatmaps_model import main_network


def main():
    dest_dir = "/home/hello/Desktop/receipt_demo"
    # resized_image
    img_ph = tf.placeholder(tf.float32, shape=[None, 600, 800, 3])
    keypoints_prediction, heatmaps_prediction, _,_ = main_network(img_ph)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "/home/hello/tf_doc_localisation/data/train_dir/model.ckpt-15816")

    for img_name in glob(dest_dir+"/*/*.jpg"):
        im = Image.open(img_name)
        resized_image = im.resize((600, 800))
        mat = np.array(resized_image)
        mat = np.expand_dims(mat, 0)
        kpnd, heatmap_nd = sess.run([keypoints_prediction, heatmaps_prediction], feed_dict={img_ph:mat})
        print (kpnd)

if __name__ == "__main__":
    main()
