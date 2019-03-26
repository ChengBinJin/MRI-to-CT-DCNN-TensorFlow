import os
import sys
import cv2
import argparse
import numpy as np
import SimpleITK as sitk
from utils import all_files_under

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--mr_path', dest='mr_path', default='../../Data/brain01/mr',
                    help='path of MR image folder')
parser.add_argument('--mask_path', dest='mask_path', default='../../Data/brain01/mask',
                    help='path of mask image folder')
parser.add_argument('--size', dest='size', default=256, type=int,
                    help='image width == height')
parser.add_argument('--delay', dest='delay', default=0, type=int,
                    help='wait delay when showing image')
parser.add_argument('--is_save', dest='is_save', default=False, action='store_true',
                    help='save N4 Bias Field Correction images')
args = parser.parse_args()


def main(mr_folder, mask_folder, size=256, is_save=False, delay=0):
    mr_paths = all_files_under(mr_folder, extension='png')
    mask_paths = all_files_under(mask_folder, extension='png')

    save_folder = None
    if is_save:
        save_folder = os.path.join(os.path.dirname(mr_folder), 'mr_correction')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    for idx, (mr_path, mask_path) in enumerate(zip(mr_paths, mask_paths)):
        print('idx: {}, MR path: {}'.format(idx, mr_path))

        ori_img, cor_img = n4itk(idx, mr_path, mask_path)
        show_imgs(ori_img, cor_img, size, delay)

        if is_save:
            save_imgs(save_folder, mr_path, cor_img)


def n4itk(idx, mr_path, mask_path):
    # Initialize reader IO
    reader = sitk.ImageFileReader()
    reader.SetImageIO("PNGImageIO")
    # Read MR image
    reader.SetFileName(mr_path)
    mr_img = reader.Execute()
    ori_img = sitk.GetArrayFromImage(mr_img)
    # Read Mask image
    reader.SetFileName(mask_path)
    mask_img = reader.Execute()

    # Convert (0, 255) to (0, 1)
    mask_img = sitk.GetArrayFromImage(mask_img)
    mask_img[mask_img > 1] = 1
    mask_img = sitk.GetImageFromArray(mask_img)

    # mask_img = sitk.OtsuThreshold(mr_img, 0, 1, 200)

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
        return ori_img, cor_img  # return origin image and corrected image
    except (RuntimeError, TypeError, NameError):
        print('[*] Catch the RuntimeError!')
        return ori_img, ori_img


def show_imgs(img1, img2, size=256, delay=0):
    canvas = np.zeros((size, 2 * size), dtype=np.uint8)
    canvas[:, :size] = img1
    canvas[:, -size:] = img2

    cv2.imshow("N4 Bias Field Correction", canvas)
    if cv2.waitKey(delay) & 0XFF == 27:
        sys.exit('[*] Esc clicked!')

def save_imgs(save_folder, mr_path, img):
    # processing
    img[img < 0] = 0
    img[img > 255] = 255
    img_int = img.astype(np.uint8)

    save_path = os.path.join(save_folder, os.path.basename(mr_path))
    cv2.imwrite(save_path, img_int)


if __name__ == '__main__':
    if not os.path.exists(args.mr_path) or not os.path.exists(args.mask_path):
        sys.exit("MR folder or Mask folder not exist!")

    main(mr_folder=args.mr_path,
         mask_folder=args.mask_path,
         size=args.size,
         is_save=args.is_save,
         delay=args.delay)
