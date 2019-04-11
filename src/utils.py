# ---------------------------------------------------------
# Python Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------

import os
import cv2
import numpy as np
import SimpleITK as sitk


def all_files_under(path, extension='png', append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def histogram(img, bins=256):
    h, w = img.shape
    hist = np.zeros(bins)
    for i in range(h):
        for j in range(w):
            a = img.item(i, j)
            hist[a] += 1

    return hist


def cumulative_histogram(hist, bins=256):
    cum_hist = hist.copy()
    for i in range(1, bins):
        cum_hist[i] = cum_hist[i-1] + cum_hist[i]

    return cum_hist


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
