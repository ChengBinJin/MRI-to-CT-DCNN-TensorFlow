# ---------------------------------------------------------
# Python Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------

import os
import numpy as np


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

