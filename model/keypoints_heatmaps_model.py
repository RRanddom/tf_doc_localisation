from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf 
import numpy as np
from model import dsnt
from model.feature_extractor import extract_features
from utils.misc import get_init_fn_for_scaffold

def main_network(images, depth_multiplier=.5, is_training=False):
    _, endpoints = extract_features(images, depth_multiplier=depth_multiplier, final_endpoint="layer_19", is_training=is_training)
    activation_maps = endpoints["layer_19"]

    with tf.variable_scope("heats_map_regression"):
        net = tf.identity(activation_maps, 'mid_layer')
        keypoints_logits = tf.layers.conv2d(net, 4, kernel_size=1, activation=None, name="pred_keypoints")

        heatmaps = []
        keypoints = []

        for i in range(4):
            heatmap, keypoint = dsnt.dsnt(keypoints_logits[..., i])
            # origin keypoint ~ [-1,1].
            keypoint = (keypoint+1)/2
            heatmaps.append(heatmap)
            keypoints.append(keypoint)

    keypoints_prediction = tf.stack(keypoints, axis=1, name="keypoints_pred")
    heatmaps_prediction = tf.add_n(heatmaps, name="heatmaps")
    
    return keypoints_prediction, heatmaps_prediction, keypoints, heatmaps

def keypoints_heatmaps_model(features, labels, mode, params=None):
    """
    """
    width, height = params["width"], params["height"]
    features.set_shape([None, height, width, 3])

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    depth_multiplier = 1
    if "depth_multiplier" in params:
        depth_multiplier = params["depth_multiplier"]

    keypoints_pred, heatmaps_pred, keypoints, heatmaps  = main_network(features, depth_multiplier, is_training)

    predictions = {
        "raw_imgs" : features,
    }

    tf.summary.image("keypoints_heatmap",  tf.expand_dims(heatmaps_pred,3))
    tf.summary.image("raw_image", features)
    
    predictions.update({"keypoints_preds" : keypoints_pred})
    predictions.update({"heatmaps_preds"  : heatmaps_pred})

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    gt_keypoints = labels
    tmp1 = gt_keypoints[:,:,0] / width
    tmp2 = gt_keypoints[:,:,1] / height 
    gt_keypoints_norm = tf.stack([tmp1, tmp2], axis=2)

    total_loss = 0
    total_mse_loss = 0
    total_reg_loss = 0
    
    tensors_to_log = {} 
    for i in range(4):
        gt_landmark_i = gt_keypoints_norm[:,i,:]
        mse_loss = tf.losses.mean_squared_error(gt_landmark_i, keypoints[i])
        reg_loss = dsnt.js_reg_loss(heatmaps[i], gt_landmark_i)
        
        total_mse_loss += mse_loss
        total_reg_loss += reg_loss
        
        tf.summary.scalar("mse_loss_for_point_"+str(i+1), mse_loss+reg_loss)
        tensors_to_log.update({"mse_loss_for_point_"+str(i+1) : mse_loss})
        total_loss += (mse_loss+reg_loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                predictions=predictions)

    global_step = tf.train.get_or_create_global_step()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    tensors_to_log.update({"total_mse_loss" : total_mse_loss})
    tensors_to_log.update({"total_reg_loss" : total_reg_loss})

    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=params["learning_rate"]).minimize(total_loss, global_step)

    train_dir = params["train_dir"]
    pretrained_model = params["pretrained_model"]
    
    scaffold = tf.train.Scaffold(init_fn=get_init_fn_for_scaffold(pretrained_model, train_dir, keywords="Mobile"))
    
    loss_tensor = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=20, formatter=lambda dicts: "\n" + ', '.join([' %s=%s' % (k, v) for k, v in dicts.items()]))

    return tf.estimator.EstimatorSpec(
            mode=mode, 
            loss=total_loss, 
            train_op=train_op,
            predictions=predictions,
            training_hooks=[loss_tensor],
            scaffold=scaffold)
