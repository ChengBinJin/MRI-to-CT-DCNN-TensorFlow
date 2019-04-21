# ---------------------------------------------------------
# Tensorflow DCNN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import sys
import cv2
import argparse
import numpy as np
# import matplotlib.pyplot as plt

from utils import all_files_under

parser = argparse.ArgumentParser(description='get_mask')
parser.add_argument('--data', dest='data', default='../../Data/brain01/raw',
                    help='dataset for making mask')
parser.add_argument('--size', dest='size', default=256, type=int, help='image width == height')
parser.add_argument('--delay', dest='delay', default=0, type=int, help='wait delay when showing image')
parser.add_argument('--task', dest='task', default='m2c', help='is m2c or c2m, default: m2c')
parser.add_argument('--is_save', dest='is_save', default=False, action='store_true',
                    help='save MR, CT, and Mask')
args = parser.parse_args()

def main(data, size=256, task='m2c', is_save=False, delay=0):
    # read file paths
    filenames = all_files_under(path=data)

    # Note
    if task.lower() == 'c2m':
        print('*' * 60)
        print('[!] Estimating C2M mask should be more improved. '
              'We recommend the task of C2M to use this function.')
        print('*' * 60)

    # Construct saving folder
    save_path = os.path.join(os.path.dirname(data), 'get_mask_' + task)
    if is_save and not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx, filename in enumerate(filenames):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        ct, mr = img[:, :size], img[:, -size:]

        # Calculate mask
        if task.lower() == 'm2c':
            mask = get_mask(mr, task=task)
        elif task.lower() == 'c2m':
            mask = get_mask(ct, task=task)
        else:
            raise NotImplementedError

        # Masked out
        masked_ct = ct & mask
        masked_mr = mr & mask
        imgs = [ct, mask, masked_ct, mr, mask, masked_mr]
        canvas = imshow(images=imgs, task=task.lower(), size=size, delay=delay)

        if is_save:
            imwrite(image=canvas, save_path=save_path, filename=filename)


def get_mask(image, task='m2c'):
    # Bilateral Filtering
    # img_blur = cv2.bilateralFilter(image, 5, 75, 75)
    img_blur = image.copy()
    th, img_thr = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_mor = img_thr.copy()

    # For loop closing
    for ksize in range(21, 3, -2):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        img_mor = cv2.morphologyEx(img_mor, cv2.MORPH_CLOSE, kernel)

    # Copy the thresholded image.
    im_floodfill = img_mor.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = img_mor.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0,0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    img_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    pre_mask = img_mor | img_floodfill_inv

    # Find the biggest contour
    mask = np.zeros((h, w), np.uint8)
    max_pix, max_cnt = 0, None
    _, contours, _ = cv2.findContours(pre_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        num_pix = cv2.contourArea(cnt)
        if num_pix > max_pix:
            max_pix = num_pix
            max_cnt = cnt

    cv2.drawContours(mask, [max_cnt], 0, 255, -1)

    if task.lower() == 'm2c':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=2)

    return mask

def imshow(images, task='m2c', size=256, delay=0):
    canvas = np.zeros((2 * size, 3 * size), np.uint8)

    if task == 'm2c':
        canvas[:size, :size] = images[3]
        canvas[:size, size:2*size] = images[4]
        canvas[:size, 2*size:] = images[5]
        canvas[size:, :size] = images[0]
        canvas[size:, size:2*size] = images[1]
        canvas[size:, 2*size:] = images[2]
    elif task == 'c2m':
        canvas[:size, :size] = images[0]
        canvas[:size, size:2*size] = images[1]
        canvas[:size, 2*size:] = images[2]
        canvas[size:, :size] = images[3]
        canvas[size:, size:2*size] = images[4]
        canvas[size:, 2*size:] = images[5]
    else:
        raise NotImplementedError


    cv2.imshow(task.upper(), canvas)

    if cv2.waitKey(delay) & 0xFF == 27:
        sys.exit('[*] Esc clicked!')

    return canvas

def imwrite(image, save_path, filename=None):
    cv2.imwrite(os.path.join(save_path, os.path.basename(filename)), image)


if __name__ == '__main__':
    if args.task != 'c2m' and args.task != 'm2c':
        sys.exit("[*] Input task is not proper!")

    main(data=args.data, size=args.size, task=args.task, delay=args.delay, is_save=args.is_save)
