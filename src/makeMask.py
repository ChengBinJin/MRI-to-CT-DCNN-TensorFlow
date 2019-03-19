import os
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import all_files_under

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--data', dest='data', default='../../Data/brain01', help='dataset for making mask')
parser.add_argument('--size', dest='size', default=256, type=int, help='image width == height')
parser.add_argument('--delay', dest='delay', default=0, type=int, help='wait delay when showing image')
parser.add_argument('--task', dest='task', default='m2c', help='task is m2c or c2m, default: m2c')
args = parser.parse_args()

def main():
    # read file paths
    filenames = all_files_under(path=args.data)

    for idx, filename in enumerate(filenames):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        ct, mr = img[:, :args.size], img[:, -args.size:]

        # Calculate mask
        if args.task == 'c2m':
            mask = find_mask(ct)
        elif args.task == 'm2c':
            mask = find_mask(mr)
        else:
            raise NotImplementedError

        # Masked out
        masked_ct = ct & mask
        masked_mr = mr & mask

        imgs = [ct, mask, masked_ct, mr, mask, masked_mr]
        plot_images(images=imgs)


def find_mask(image):
    # Bilateral Filtering
    img_blur = cv2.bilateralFilter(image, 9, 75, 75)
    th, img_thr = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_mor = img_thr.copy()

    # For loop closing
    for ksize in range(15, 3, -2):
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

    if args.task == 'm2c':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=2)

    return mask

def plot_images(images):
    canvas = np.zeros((2 * args.size, 3 * args.size), np.uint8)

    canvas[:args.size, :args.size] = images[0]
    canvas[:args.size, args.size:2*args.size] = images[1]
    canvas[:args.size, 2*args.size:] = images[2]
    canvas[args.size:, :args.size] = images[3]
    canvas[args.size:, args.size:2*args.size] = images[4]
    canvas[args.size:, 2*args.size:] = images[5]

    cv2.imshow('img | mask | masked', canvas)

    if cv2.waitKey(args.delay) & 0xFF == 27:
        sys.exit('[*] Esc clicked!')


if __name__ == '__main__':
    main()
