import os
import sys
import random
import numpy as np
from glob import glob
from PIL import Image, ImageDraw, ImageFilter

image_width, image_height = 600, 800

def shift_image(image_nd, dx, dy):
    image_nd = np.roll(image_nd, dy, axis=0)
    image_nd = np.roll(image_nd, dx, axis=1)
    if dy>0:
        image_nd[:dy, :] = 0
    else:
        image_nd[dy:, :] = 0
    if dx>0:
        image_nd[:, :dx] = 0
    else:
        image_nd[:, dx:] = 0
    return image_nd


def extract_receipt(receipt_image, points):
    points_nd = np.array(points)
    min_x, max_x = np.min(points_nd[:,0]), np.max(points_nd[:,0])
    min_y, max_y = np.min(points_nd[:,1]), np.max(points_nd[:,1])

    cropped = receipt_image.crop(box=[min_x,min_y,max_x,max_y])

    points_nd[:,0] -= min_x
    points_nd[:,1] -= min_y

    points = list(points_nd)
    points = [tuple(point) for point in points]

    return cropped,points


def random_resize_receipt(receipt_image, points):
    """
    """
    receipt_width, receipt_height = receipt_image.size

    the_origin_receipt_factor = max(1.0*receipt_width/image_width, 1.0*receipt_height/image_height)
    # factor = [.4, .9]
    the_factor_we_expect = np.random.rand() * .5 + .5 # factor. [.5, 1.0]
    the_factor_we_should_apply = the_factor_we_expect/the_origin_receipt_factor

    resize_width, resize_height = int(receipt_width*the_factor_we_should_apply) , int(receipt_height*the_factor_we_should_apply)
    receipt_image_resized = receipt_image.resize((resize_width, resize_height))

    points_nd = (np.array(points) * the_factor_we_should_apply).astype('int32')
    points = list(points_nd)
    points = [tuple(point) for point in points]

    return receipt_image_resized,points

def random_flip_receipt(receipt_image, points):
    receipt_width, receipt_height = receipt_image.size

    method = random.choice([Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM])
    receipt_image_flipped = receipt_image.transpose(method)
    points_nd = np.array(points)

    # p1, p2, p3, p4 = points
    if method == Image.FLIP_LEFT_RIGHT:
        points_nd[:,0] = receipt_width - points_nd[:,0]
        points = list(points_nd)
        points = [tuple(point) for point in points]
        p1,p2,p3,p4 = points

        points = [p2, p1, p4, p3]

    elif method == Image.FLIP_TOP_BOTTOM:
        points_nd[:,1] = receipt_height - points_nd[:,1]
        points = list(points_nd)
        points = [tuple(point) for point in points]
        p1,p2,p3,p4 = points

        points = [p3, p4, p1, p2]
    
    return receipt_image_flipped, points


def random_synthesis_v2(raw_image, points, bg_image):
    #################################################
    #                                               #
    #       pt1 ------------------pt2               #
    #          |                  |                 #
    #          |                  |                 #
    #          |                  |                 #
    #          |                  |                 #
    #       pt3-------------------pt4               #
    #                                               #
    #################################################
    bg_image_cp = bg_image.copy()
    raw_img_pil = raw_image.convert("RGBA")
    width, height = raw_img_pil.size

    pt1, pt2, pt3, pt4 = points
    padding = 3
    pt1 = (pt1[0]+padding, pt1[1]+padding)
    pt2 = (pt2[0]-padding, pt2[1]+padding)
    pt3 = (pt3[0]+padding, pt3[1]-padding)
    pt4 = (pt4[0]-padding, pt4[1]-padding)
    points = [pt1, pt2, pt3, pt4]

    mask = Image.new('L', raw_img_pil.size, 0)
    ImageDraw.Draw(mask).polygon([pt1, pt2, pt4, pt3], outline=1, fill=1)
    mask_nd = np.array(mask)    

    new_im_nd = np.empty((height, width, 4), dtype='uint8')
    new_im_nd[:,:,:3] = np.array(raw_img_pil)[:,:,:3]
    new_im_nd[:,:,3] = mask_nd * 255

    receipt_image_pil = Image.fromarray(new_im_nd, 'RGBA')

    receipt_image, points = extract_receipt(receipt_image_pil, points)

    resize_prob = .9
    should_resize = (np.random.rand() < resize_prob)
    if should_resize:
        receipt_image, points = random_resize_receipt(receipt_image, points)
    
    # should rotate.
    # it is difficult. emmm.
    
    flip_prob = .5
    should_flip = (np.random.rand() < flip_prob)

    if should_flip:
        receipt_image, points = random_flip_receipt(receipt_image, points)

    receipt_w, receipt_h = receipt_image.size    
    
    coord_x = random.randint(0, (image_width - receipt_w))
    coord_y = random.randint(0, (image_height - receipt_h))
    bg_image_cp.paste(receipt_image, (coord_x, coord_y), receipt_image)
    systhesis_img = bg_image_cp.filter(ImageFilter.GaussianBlur(radius=.5))

    points_nd = np.array(points)
    points_nd[:,0] += coord_x
    points_nd[:,1] += coord_y
    points = list(points_nd)
    points = [tuple(point) for point in points]

    return systhesis_img, points


def rotate_point(point, center, angle):
    """ let's roate the origin point around the center with given angle.
    """
    center_x,center_y = center
    px, py = point

    new_x = center_x + np.math.cos(angle) * (px - center_x) - np.math.sin(angle) * (py - center_y)
    new_y = center_y + np.math.sin(angle) * (px - center_x) + np.math.cos(angle) * (py - center_y)

    return (int(new_x), int(new_y))

def random_synthesis_rec_with_bgimg(rec_image, bg_image):
    # def random_synthesis_v2(raw_image, points, bg_image):
    """ Systhesis rec_image & bg_image. Apply some random transformation to rec_image, and paste the rec_image to bg_image.

    Args:
        rec_image:
        bg_image:
    Return:
        systhesis_img: PIL Image with RGB mode.
        points:      : 4 corner points.
    """ 
    rec_width, rec_height = rec_image.size
    rotate_prob = 1.0
    should_rotate = (np.random.rand()<rotate_prob)

    p1 = (0,0)
    p2 = (rec_width, 0)
    p3 = (0, rec_height)
    p4 = (rec_width, rec_height)
    points = [p1,p2,p3,p4]

    if should_rotate:
        rot_degree = np.random.rand() * 90 - 45.0 # [-45, 45].
        rec_image = rec_image.rotate(rot_degree, expand=True) #

        np_angle = np.deg2rad(rot_degree)

        center_p = tuple(np.array(points).mean(axis=0))
        points = [rotate_point(point, center_p, -np_angle) for point in points]

        points_nd = np.array(points)
        x_max, x_min = points_nd[:,0].max(), points_nd[:,0].min()
        y_max, y_min = points_nd[:,1].max(), points_nd[:,1].min()
        points_nd[:,0] -= x_min
        points_nd[:,1] -= y_min
        
        points = [tuple(point) for point in list(points_nd)]
    
    receipt_image_pil = rec_image

    rec_image, points = extract_receipt(rec_image, points)

    rec_image, points = random_resize_receipt(rec_image, points)
    
    flip_prob = .5
    should_flip = (np.random.rand() < flip_prob)

    if should_flip:
        rec_image, points = random_flip_receipt(rec_image, points)

    receipt_w, receipt_h = rec_image.size    
    
    coord_x = random.randint(0, (image_width - receipt_w))
    coord_y = random.randint(0, (image_height - receipt_h))
    bg_image_cp = bg_image.copy()
    bg_image_cp.paste(rec_image, (coord_x, coord_y), rec_image)
    systhesis_img = bg_image_cp.filter(ImageFilter.GaussianBlur(radius=.5))

    points_nd = np.array(points)
    points_nd[:,0] += coord_x
    points_nd[:,1] += coord_y
    points = list(points_nd)
    points = [tuple(point) for point in points]

    return systhesis_img, points


def gen_synthesis_data_v1(dest_path, iteration = 1):
    """
    crop real receipt images and blend with background image.
    """
    bg_images_path1 = 'raw_data/ocr_train_image_background'
    bg_images_path2 = 'raw_data/other_bgs'

    images_path = 'raw_data/ocr_image'
    annot_path = 'raw_data/ocr_annot'

    if not os.path.exists(images_path):
        raise ValueError("directory {} not exists".format(annot_path))

    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    
    counter = 0
    images_ct = len(glob(os.path.join(images_path, '*.jpg')))

    all_bg_imgs = glob(os.path.join(bg_images_path1, '*.jpg')) + glob(os.path.join(bg_images_path2, '*.jpg'))
    bg_images_ct = len(all_bg_imgs)
    total_images_ct = iteration * bg_images_ct * images_ct

    for _iter in range(iteration):

        for bg_image_name in all_bg_imgs:
            bg_image = Image.open(bg_image_name)
            bg_base_name = os.path.basename(bg_image_name).split('.')[0]

            for img_name in glob(os.path.join(images_path, '*.jpg')):
                image_base_name = os.path.basename(img_name).split('.')[0]
                raw_image = Image.open(img_name)
                pts = open(os.path.join(annot_path, image_base_name+'.txt'), 'r').readline().strip().split(',')
                pts = [int(pt) for pt in pts]

                points = [(pts[0], pts[1]), (pts[2], pts[3]), (pts[4], pts[5]), (pts[6], pts[7])]

                synthesis_name = '_'.join([bg_base_name, image_base_name, str(_iter)])

                synthesis_image, points_altered = random_synthesis_v2(raw_image, points, bg_image)
                synthesis_image.save(os.path.join(dest_path, synthesis_name+'.jpg'))

                pts_altered = []
                for p in points_altered:
                    pts_altered.append(str(p[0]))
                    pts_altered.append(str(p[1]))

                with open(os.path.join(dest_path, synthesis_name +'.txt'), 'w+') as txt_f:
                    txt_f.write(','.join(pts_altered))

                sys.stdout.write('\r>> synthesis image %d/%d' % (
                    counter + 1, total_images_ct))
                counter += 1
                sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()

def gen_synthesis_data_v2(dest_path, iteration = 1):
    """
    blend rectangle images with background images.
    """
    bg_images_path1 = 'raw_data/ocr_train_image_background'
    bg_images_path2 = 'raw_data/other_bgs'
    rec_images_path = 'raw_data/rec_imgs'

    if not os.path.exists(rec_images_path):
        raise ValueError("directory {} not exists".format(rec_images_path))
    
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    
    all_bg_images = glob(os.path.join(bg_images_path1, '*.jpg')) + glob(os.path.join(bg_images_path2, "*.jpg"))
    all_rec_images = glob(os.path.join(rec_images_path, '*.jpg'))

    total_ct = iteration * len(all_rec_images) * len(all_bg_images)
    counter = 0

    for _iter in range(iteration):
        for bg_image_name in all_bg_images:
            bg_image = Image.open(bg_image_name)
            bg_base_name = os.path.basename(bg_image_name).split('.')[0]

            for rec_image_name in all_rec_images:
                image_base_name = os.path.basename(rec_image_name).split('.')[0]
            
                raw_rec_image = Image.open(rec_image_name).convert("RGBA")
                synthesis_image, points = random_synthesis_rec_with_bgimg(raw_rec_image, bg_image)

                pts = []
                for p in points:
                    pts.append(str(p[0]))
                    pts.append(str(p[1]))

                synthesis_name = '_'.join([bg_base_name, image_base_name, str(_iter)])
                synthesis_image.save(os.path.join(dest_path, synthesis_name+'.jpg'))
                with open(os.path.join(dest_path, synthesis_name +'.txt'), 'w+') as txt_f:
                    txt_f.write(','.join(pts))

                sys.stdout.write('\r>> synthesis image %d/%d' % (
                    counter + 1, total_ct))
                counter += 1
                sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':
    gen_synthesis_data_v1('data/receipts', iteration=1)
    gen_synthesis_data_v2('data/receipts', iteration=1)    
