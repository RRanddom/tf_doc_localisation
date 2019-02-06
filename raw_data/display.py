from PIL import Image, ImageDraw
import numpy as np
from glob import glob
import random
import os

def main():
    iphone_dir = 'ocr_image_iphone'
    iphone_annot_dir = 'ocr_annot_iphone'
    huawei_dir = 'ocr_image_huawei'
    huawei_annot_dir = 'ocr_annot_huawei'

    dest_dir = 'display'

    iphone_annots = glob(iphone_annot_dir+ '/*.txt')
    random.shuffle(iphone_annots)
    iphone_annots = iphone_annots[:20]

    huawei_annots = glob(huawei_annot_dir+ '/*.txt')
    random.shuffle(huawei_annots)
    huawei_annots = huawei_annots[:20]

    def plot_and_draw(annots, src_image_dir, dest_image_dir):
        for annot_file in annots:
            annot_base_name = os.path.basename(annot_file).split('.')[0]
            image_name = os.path.join(src_image_dir, annot_base_name+'.jpg')
            if os.path.exists(image_name):
                img_pil = Image.open(image_name)
                drawer = ImageDraw.Draw(img_pil)
                pts = [int(pt) for pt in open(annot_file, 'r').readline().strip().split(',')]
                pt1, pt2, pt3, pt4 = (pts[0], pts[1]), (pts[2], pts[3]), (pts[4], pts[5]), (pts[6], pts[7])
                drawer.polygon([pt1, pt2, pt4, pt3], outline=(0,255,0))
                dest_full_path = os.path.join(dest_image_dir, annot_base_name+'.jpg')
                img_pil.save(dest_full_path)

    plot_and_draw(iphone_annots, iphone_dir, os.path.join(dest_dir, 'iphone'))
    plot_and_draw(huawei_annots, huawei_dir, os.path.join(dest_dir, 'huawei'))


def crop():
    import numpy
    from PIL import Image, ImageDraw

    # read image as RGB and add alpha (transparency)
    im = Image.open("ocr_image_huawei/receipt_375.jpg").convert("RGBA")
    # convert to numpy (for convenience)
    imArray = numpy.asarray(im)
    # create mask
    polygon = [(135+4,153+4),(511-2,145+4),(622-2,779-2),(184+4,825-2)]
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = numpy.array(maskIm)

    # assemble new image (uint8: 0-255)
    newImArray = numpy.empty(imArray.shape,dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:,:,:3] = imArray[:,:,:3]

    # transparency (4th column)
    newImArray[:,:,3] = mask*255

    # back to Image from numpy
    newIm = Image.fromarray(newImArray, "RGBA")

    bg = Image.open('ocr_train_image_background/receipt_508.jpg').convert("RGBA")

    bg.paste(newIm, (0, 0), newIm)
    # result= Image.blend(bg, newIm, alpha=0.0)
    bg.save("out.png")

def random_merge():
    iphone_dir = 'ocr_image_iphone'
    iphone_annot_dir = 'ocr_annot_iphone'
    huawei_dir = 'ocr_image_huawei'
    huawei_annot_dir = 'ocr_annot_huawei'
    background_dir = 'ocr_train_image_background'

    dest_dir = 'merge'

    iphone_annots = glob(iphone_annot_dir+ '/*.txt')
    random.shuffle(iphone_annots)
    iphone_annots = iphone_annots[:20]

    huawei_annots = glob(huawei_annot_dir+ '/*.txt')
    random.shuffle(huawei_annots)
    huawei_annots = huawei_annots[:20]

    bg_images = glob(background_dir+'/*.jpg')

    def merge(annots, src_image_dir, bg_images):
        for bg_image in bg_images:
            bg_base_name = os.path.basename(bg_image).split('.')[0]

            for annot_file in annots:
                bg_pil = Image.open(bg_image)

                annot_base_name = os.path.basename(annot_file).split('.')[0]
                image_name = os.path.join(src_image_dir, annot_base_name+'.jpg')
                if os.path.exists(image_name):
                    img_pil = Image.open(image_name).convert("RGBA")
                    mask_pil = Image.new('L', img_pil.size, 0)

                    pts = [int(pt) for pt in open(annot_file, 'r').readline().strip().split(',')]
                    pt1, pt2, pt3, pt4 = (pts[0]+3, pts[1]+3), (pts[2]-3, pts[3]+3), (pts[4]+3, pts[5]-3), (pts[6]-3, pts[7]-3)
                    ImageDraw.Draw(mask_pil).polygon([pt1, pt2, pt4, pt3], outline=1, fill=1)
                    mask = np.array(mask_pil)

                    imArray = np.array(img_pil)
                    # assemble new image (uint8: 0-255)
                    newImArray = np.empty(imArray.shape,dtype='uint8')

                    # colors (three first columns, RGB)
                    newImArray[:,:,:3] = imArray[:,:,:3]

                    # transparency (4th column)
                    newImArray[:,:,3] = mask*255

                    newIm = Image.fromarray(newImArray, "RGBA")
                    bg_pil.paste(newIm, (0,0), newIm)
                    
                    img_name = bg_base_name+'_'+annot_base_name+'.jpg'
                    
                    bg_pil.convert('RGB').save(os.path.join('merge', img_name))

                    #drawer.polygon([pt1, pt2, pt4, pt3], outline=(0,255,0))
    
    # merge(iphone_annots, iphone_dir, bg_images)
    merge(huawei_annots, huawei_dir, bg_images)


if __name__ == '__main__':
    random_merge()
    # main()
