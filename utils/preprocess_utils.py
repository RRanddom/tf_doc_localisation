import tensorflow as tf

#################################################
#                                               #
#       pt1 ------------------pt2               #
#          |                  |                 #
#          |                  |                 #
#          |                  |                 #
#          |                  |                 #
#       pt3-------------------pt4               #
#                                               #
#                                               #
#  points_tensor = array([pt1, pt2, pt3, pt4])  #
#                                               #
#################################################



def get_bbox_from_points(points_tensor):
    """ get the bounding box from 4 points.

    Args:
        points_tensor, shape of [4, 2], 4 points.
    
    Returns:
        bbox. shape of [4,], represents. [center_x, center_y, width, height]
    """

    center_x = tf.reduce_mean(points_tensor[:,0])
    center_y = tf.reduce_mean(points_tensor[:,1])
    width  = tf.reduce_max(points_tensor[:,0]) - tf.reduce_min(points_tensor[:,0])
    height = tf.reduce_max(points_tensor[:,1]) - tf.reduce_min(points_tensor[:,1])

    return tf.stack([center_x, center_y, width, height])

def random_left_right_flip(image_tensor, points_tensor, image_width, image_height, prob=.2):
    """ Randomly left_right flips the image_tensor ,along with the points

    Args:
        image_tensor, tensor of image, shape = [height, width, 3]
        points_tensor, tensor of points. shape = [4, 2],
        image_width, float or tensor type.
        image_height, float or tensor type.
        prob, float point number ~[0,1]. 
    
    Returns:
        image_tensor_output: same format as image_tensor
        points_tensor_output: same format as points_tensor
    """
    random_value = tf.random_uniform([])
    is_flipped = tf.less_equal(random_value, prob)
    
    def flip():
        image_tensor_reversed = tf.reverse_v2(image_tensor, [1])

        x_coords = image_width - points_tensor[:,0]
        y_coords = points_tensor[:,1]
        tmp = tf.stack([x_coords, y_coords], axis=1)
        p1 = tmp[0,:]
        p2 = tmp[1,:]
        p3 = tmp[2,:]
        p4 = tmp[3,:]
        points_reversed = tf.stack([p2,p1,p4,p3])

        return image_tensor_reversed, points_reversed

    return tf.cond(is_flipped, flip, lambda: (image_tensor, points_tensor))


def random_up_down_flip(image_tensor, points_tensor, image_width, image_height, prob=.2):

    """ Randomly up_down flips the image_tensor ,along with the points

    Args:
        image_tensor, tensor of image, shape = [height, width, 3]
        points_tensor, tensor of points. shape = [4, 2],
        image_width, float or tensor type.
        image_height, float or tensor type.
        prob, float point number ~[0,1]. 
    
    Returns:
        image_tensor_output: same format as image_tensor
        points_tensor_output: same format as points_tensor
    """
    random_value = tf.random_uniform([])
    is_flipped = tf.less_equal(random_value, prob)
    
    def flip():
        image_tensor_reversed = tf.reverse_v2(image_tensor, [0])

        x_coords = points_tensor[:,0]
        y_coords = image_height - points_tensor[:,1]
        tmp = tf.stack([x_coords, y_coords], axis=1)
        p1 = tmp[0,:]
        p2 = tmp[1,:]
        p3 = tmp[2,:]
        p4 = tmp[3,:]
        points_reversed = tf.stack([p3,p4,p1,p2])

        return image_tensor_reversed, points_reversed

    return tf.cond(is_flipped, flip, lambda: (image_tensor, points_tensor))
