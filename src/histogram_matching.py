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

from utils import all_files_under, histogram, cumulative_histogram

parser = argparse.ArgumentParser(description='histogram_matching')
parser.add_argument('--data', dest='data', default='../../Data/brain01/raw',
                    help='dataset for histogram matching')
parser.add_argument('--temp_id', dest='temp_id', default=2, type=int, help='template image id of the histogram matching')
parser.add_argument('--size', dest='size', default=256, type=int, help='image width == height')
parser.add_argument('--delay', dest='delay', default=0, type=int, help='wait delay when showing image')
parser.add_argument('--is_save', dest='is_save', default=False, action='store_true', help='save MR image')
args = parser.parse_args()

def main(data, size, temp_id, is_save=False, delay=0):
    filenames = all_files_under(data, extension='png')

    save_folder = os.path.join(os.path.dirname(data), 'hist_match')
    if is_save and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # read teamplate image
    temp_filename = filenames[temp_id]
    ref_img = cv2.imread(temp_filename, cv2.IMREAD_GRAYSCALE)
    ref_img = ref_img[:, -size:].copy()

    for idx, filename in enumerate(filenames):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        mr_img = img[:, -size:]

        mr_img_af = histogram_matching(mr_img, ref_img)
        canvas = imshow(mr_img, mr_img_af, size=size, delay=delay)

        if is_save:
            imwrite(canvas, save_folder, filename)

def imwrite(img, save_folder, filename):
    cv2.imwrite(os.path.join(save_folder, os.path.basename(filename)), img)


def imshow(img1, img2, size=256, delay=0, margin=5):
    canvas = 255 * np.ones((size, 2*size + margin), dtype=np.uint8)

    canvas[:, :size] = img1
    canvas[:, -size:] = img2

    cv2.imshow('Histogram matching', canvas)
    if cv2.waitKey(delay) & 0xff == 27:
        sys.exit('[*] Esc clicked!')

    return canvas


def histogram_matching(img, ref, bins=256):
    assert img.shape == ref.shape

    result = img.copy()
    h, w = img.shape
    pixels = h * w

    # histogram
    hist_img = histogram(img)
    hist_ref = histogram(ref)
    # cumulative histogram
    cum_img = cumulative_histogram(hist_img)
    cum_ref = cumulative_histogram(hist_ref)
    # normalization
    prob_img = cum_img / pixels
    prob_ref = cum_ref / pixels

    new_values = np.zeros(bins)
    for a in range(bins):
        j = bins - 1
        while True:
            new_values[a] = j
            j = j - 1

            if j < 0 or prob_img[a] >= prob_ref[j]:
                break

    for i in range(h):
        for j in range(w):
            a = img.item(i, j)
            b = new_values[a]
            result.itemset((i, j), b)

    return result


if __name__ == '__main__':
    main(data=args.data, size=args.size, temp_id=args.temp_id, is_save=args.is_save, delay=args.delay)
