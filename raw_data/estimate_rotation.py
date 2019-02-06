import six
import os
import numpy as np
import cv2
from PIL import Image
from glob import glob
from scipy.ndimage import filters, interpolation, morphology, measurements, minimum

def estimate_skew_angle(raw_image):
    """
    estimate image rotation
    """

    def resize_image(image, scale):
        height, width = image.shape[:2]
        f = 1.0 * scale / min(height, width)
        return cv2.resize(image, (0,0), fx=f, fy=f)

    image_resized = resize_image(raw_image, scale=800)

    #cv2.imshow('image', image_resized)
    #cv2.waitKey(20)
    image = image_resized - image_resized.min()
    image = image / image.max()
    m = interpolation.zoom(image, .5)
    m = filters.percentile_filter(m, 80, size=(20,2))
    m = filters.percentile_filter(m, 80, size=(2,20))
    m = interpolation.zoom(m, 2.0)

    w,h = min(image.shape[1],m.shape[1]),min(image.shape[0],m.shape[0])
    flat = np.clip(image[:h,:w]-m[:h,:w]+1,0,1)
    d0,d1 = flat.shape
    o0,o1 = int(0.1*d0),int(0.1*d1)
    flat = np.amax(flat)-flat
    flat -= np.amin(flat)
    est = flat[o0:d0-o0,o1:d1-o1]
    angles = range(-20,20)
    estimates = []

    for a in angles:
        roest =interpolation.rotate(est,a,order=0,mode='constant')
        v = np.mean(roest,axis=1)
        v = np.var(v)
        estimates.append((v,a))

    _,degree = max(estimates)

    return degree
    
def main():
    all_images = glob("ocr_train_image_huawei/*.jpg")
    for image_name in all_images:
        raw_image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2GRAY)
        degree = estimate_skew_angle(raw_image)
        im = Image.open(image_name)
        im = im.rotate(degree)
        imgnd = np.array(im)
        cv2.imshow('after_rotate', imgnd[...,0])
        cv2.waitKey(20)


if __name__ == '__main__':
    main()

