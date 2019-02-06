import cv2
from glob import glob
import os 
import numpy as np
import argparse

refPt = []

def click_and_crop(event, x, y, flags, param):
    global refPt

    colors = [(0,255,77), (255,0,0), (127,89,99), (0,0,255)]

    if event == cv2.EVENT_LBUTTONDOWN:
        
        if len(refPt) == 4:
            return

        if len(refPt):
            prevPt = refPt[-1]
            if abs(x-prevPt[0])<=3 and abs(y-prevPt[1])<=3:
                # my mouse does not work well in Ubuntu!!!
                print ("double click error")
                return

        # else:
        current_index = len(refPt)
        the_color = colors[current_index]
        
        refPt.append((x, y))
        cv2.circle(image, (x,y), 5, the_color, -1)

        if len(refPt) == 4:
            cv2.line(image, refPt[0], refPt[1], (22,22,22), 5)
            cv2.line(image, refPt[1], refPt[3], (22,22,22), 5)
            cv2.line(image, refPt[0], refPt[2], (22,22,22), 5)
            cv2.line(image, refPt[2], refPt[3], (22,22,22), 5)


images_dir = 'ocr_train_iphone'
annots_dir = 'ocr_annot_iphone'


all_images = sorted(glob(os.path.join(images_dir, '*.jpg')))

# image = cv2.imread(args['image'])
# image_name = args['image']

current_image_index = 0
current_image_name = all_images[current_image_index]
image = cv2.imread(current_image_name)
image = cv2.resize(image, (0,0), fx=.75, fy=.75)
clone = image.copy()

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

while True:
    
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'): # reset
        refPt = []
        image = clone.copy()

    if key == ord('s'): # save the annotation

        if len(refPt) != 4:
            print ("cannot save")
        else:
            points = []
            for p in refPt:
                p0 = int(p[0] / .5)
                p1 = int(p[1] / .5)
                points.append(str(p0))
                points.append(str(p1))

            base_name = os.path.basename(current_image_name).split('.')[0]
            with open(os.path.join(annots_dir, base_name+'.txt'), 'w+') as f:
                f.write(','.join(points))
            
            if current_image_index >= len(all_images):
                break
            else:
                refPt = []
                current_image_index += 1
                current_image_name  = all_images[current_image_index]
                image = cv2.imread(current_image_name)
                image = cv2.resize(image, (0,0), fx=.5, fy=.5)
                clone = image.copy()
                print ("current index:{}".format(current_image_index))
        # save the annotatopn        

    if key == ord('n'):
        if len(refPt) != 0:
            print ("clear current state!")
        else:
            current_image_index += 1
            current_image_name  = all_images[current_image_index]
            image = cv2.imread(current_image_name)
            image = cv2.resize(image, (0,0), fx=.5, fy=.5)
            clone = image.copy()
            print ("current index:{}".format(current_image_index))

cv2.destroyAllWindows()
