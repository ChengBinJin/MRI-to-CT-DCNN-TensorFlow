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
import SimpleITK as sitk
from utils import all_files_under

parser = argparse.ArgumentParser(description='n4itk')
parser.add_argument('--data', dest='data', default='../../Data/brain01/raw', help='data path of the n4 bias correction')
# parser.add_argument('--mask_path', dest='mask_path', default='../../Data/brain01/mask',
#                     help='path of mask image folder')
parser.add_argument('--size', dest='size', default=256, type=int,
                    help='image width == height')
parser.add_argument('--delay', dest='delay', default=0, type=int,
                    help='wait delay when showing image')
parser.add_argument('--is_save', dest='is_save', default=False, action='store_true',
                    help='save N4 Bias Field Correction images')
args = parser.parse_args()


def main(data, size=256, is_save=False, delay=0):
    save_folder = os.path.join(os.path.dirname(data), 'N4_bias_correction')
    if is_save and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    filenames = all_files_under(data, extension='png')
    for idx, filename in enumerate(filenames):
        print('idx: {}, filename: {}'.format(idx, filename))
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        mr_img = img[:, -size:]

        ori_img, cor_img = n4itk(mr_img)
        canvas = imshow(ori_img, cor_img, size, delay)

        if is_save:
            imwrite(canvas, save_folder, filename)


def n4itk(img):
    ori_img = img.copy()
    mr_img = sitk.GetImageFromArray(img)
    mask_img = sitk.OtsuThreshold(mr_img, 0, 1, 200)

    # Convert to sitkFloat32
    mr_img = sitk.Cast(mr_img, sitk.sitkFloat32)
    # N4 bias field correction
    num_fitting_levels = 4
    num_iters = 200
    try:
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([num_iters] * num_fitting_levels)
        cor_img = corrector.Execute(mr_img, mask_img)
        cor_img = sitk.GetArrayFromImage(cor_img)

        cor_img[cor_img<0], cor_img[cor_img>255] = 0, 255
        cor_img = cor_img.astype(np.uint8)
        return ori_img, cor_img  # return origin image and corrected image
    except (RuntimeError, TypeError, NameError):
        print('[*] Catch the RuntimeError!')
        return ori_img, ori_img


def imshow(img1, img2, size=256, delay=0):
    canvas = np.zeros((size, 2 * size), dtype=np.uint8)
    canvas[:, :size] = img1
    canvas[:, -size:] = img2

    cv2.imshow("N4 Bias Field Correction", canvas)
    if cv2.waitKey(delay) & 0XFF == 27:
        sys.exit('[*] Esc clicked!')

    return canvas

def imwrite(img, save_folder, filename):
    # processing
    img[img < 0] = 0
    img[img > 255] = 255
    img_int = img.astype(np.uint8)
    cv2.imwrite(os.path.join(save_folder, os.path.basename(filename)), img_int)


if __name__ == '__main__':
    main(data=args.data, size=args.size, is_save=args.is_save, delay=args.delay)
