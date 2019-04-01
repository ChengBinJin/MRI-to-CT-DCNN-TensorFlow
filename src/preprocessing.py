import os
import sys
import argparse
import cv2
import numpy as np

from n4itk import n4itk
from histogram_matching import histogram_matching
from get_mask import get_mask
from utils import all_files_under

paser = argparse.ArgumentParser(description='')
paser.add_argument('--data', dest='data', default='../../Data/brain01/raw', help='data path for preprocessing')
paser.add_argument('--temp_id', dest='temp_id', default=0, type=int, help='template id of the histogram matching')
paser.add_argument('--size', dest='size', default=256, type=int, help='image wdith == height')
paser.add_argument('--delay', dest='delay', default=0, type=int, help='interval time when showing image')
paser.add_argument('--is_save', dest='is_save', default=False, action='store_true', help='save image')
args = paser.parse_args()

def main(data, temp_id, size=256, delay=0, is_save=False):
    save_folder = os.path.join(os.path.dirname(data), 'preprocessing')
    if is_save and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # read all files paths
    filenames = all_files_under(data, extension='png')

    # read template image
    temp_filename = filenames[temp_id]
    ref_img = cv2.imread(temp_filename, cv2.IMREAD_GRAYSCALE)
    ref_img = ref_img[:, -size:].copy()
    _, ref_img = n4itk(ref_img)  # N4 bias correction for the reference image

    for idx, filename in enumerate(filenames):
        print('idx: {}, filename: {}'.format(idx, filename))

        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        ct_img = img[:, :size]
        mr_img = img[:, -size:]

        # N4 bias correction
        ori_img, cor_img = n4itk(mr_img)
        # Dynamic histogram matching between two images
        his_mr = histogram_matching(cor_img, ref_img)
        # Mask estimation based on Otsu auto-thresholding
        mask=get_mask(his_mr, task='m2c')
        # Masked out
        masked_ct = ct_img & mask
        masked_mr = his_mr & mask
        canvas = imshow(ori_img, cor_img, his_mr, masked_mr, mask, ct_img, masked_ct, size=size, delay=delay)

        if is_save:
            cv2.imwrite(os.path.join(save_folder, os.path.basename(filename)), canvas)

def imshow(ori_mr, cor_mr, his_mr, masked_mr, mask, ori_ct, masked_ct, size=256, delay=0, himgs=2, wimgs=5, margin=5):
    canvas = 255 * np.ones((himgs * size + (himgs-1) * margin, wimgs * size + (wimgs-1) * margin), dtype=np.uint8)

    first_rows = [ori_mr, cor_mr, his_mr, masked_mr, mask]
    second_rows = [ori_ct, 255*np.ones(ori_ct.shape), 255*np.ones(ori_ct.shape), masked_ct, mask]
    for idx in range(len(first_rows)):
        canvas[:size, idx*(margin+size):idx*(margin+size)+size] = first_rows[idx]
        canvas[-size:, idx*(margin+size):idx*(margin+size)+size] = second_rows[idx]

    cv2.imshow("N4 Bias Field Correction", canvas)
    if cv2.waitKey(delay) & 0XFF == 27:
        sys.exit('[*] Esc clicked!')

    return canvas


if __name__ == '__main__':
    main(data=args.data, temp_id=args.temp_id, size=args.size, delay=args.delay, is_save=args.is_save)
