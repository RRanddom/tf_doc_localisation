import tensorflow as tf 

def get_init_fn_for_scaffold(pretrained_path, model_dir, keywords=None, exclude_vars=[]):
    """ load pretrained model params to the graph.
    
    Args:
        pretrained_path: pretrained model ckpt file path
        model_dir:       trainining dir
        keywords:        ....
        exclude_vars:    dont need it now

    Returns:
        callback function to init the graph
    """
    if tf.train.latest_checkpoint(model_dir):
        tf.logging.info("Ignore pretrained path because a checkpoint file already exists")
        return None

    variables_to_restore = tf.trainable_variables()
    if keywords:
        variables_to_restore = [_var for _var in variables_to_restore if _var.name.startswith(keywords)]

    if tf.gfile.IsDirectory(pretrained_path):
        pretrained_path = tf.train.latest_checkpoint(pretrained_path)
    
    tf.logging.info("Fine tuning from %s" %(pretrained_path))

    if not variables_to_restore:
        raise ValueError("variables to restore cannot be empty.")
   
    saver = tf.train.Saver(variables_to_restore, reshape=False)
    saver.build()

    print ("pretrained_path:{}".format(pretrained_path))
    def callback(scaffold, session):
        saver.restore(session, pretrained_path)
    
    return callback