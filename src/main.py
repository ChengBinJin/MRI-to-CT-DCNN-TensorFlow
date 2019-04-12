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

from dataset import Dataset
# from model import Model
# from solver import Solver

parser = argparse.ArgumentParser(description='main')
parser.add_argument('--gpu_index', dest='gpu_index', default='0',
                    help='gpu index if you have multiple gpus, default: 0')
parser.add_argument('--is_train', dest='is_train', default=False, action='store_true',
                    help='train mode, default: False')
parser.add_argument('--batch_size', dest='batch_size', default=48, type=int,
                    help='batch size for one iteration')
parser.add_argument('--dataset', dest='dataset', default='brain01',
                    help='dataset name, default: brain01')
parser.add_argument('--learning_rate', dest='learning_rate', default=2e-4, type=float,
                    help='learning rate, default: 2e-4')
parser.add_argument('--beta1', dest='beta1', default=0.5, type=float,
                    help='momentum term of Adam, default: 0.5')
parser.add_argument('--epoch', dest='epoch', default=600, type=int,
                    help='number of epochs, default: 600')
parser.add_argument('--print_freq', dest='print_freq', default=100, type=int,
                    help='print frequency for loss, default: 100')
parser.add_argument('--load_model', dest='load_model', default=None,
                    help='folder of saved model that you wish to continue training, '
                         '(e.g., 20190411-2217), default: None')
args = parser.parse_args()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index

    num_cross_vals = 6
    for idx_cross_val in range(num_cross_vals):
        data = Dataset(args.dataset, num_cross_vals, 5)

        num_iters = int(args.epoch * data.num_train / args.batch_size)
        for iter_time in range(num_iters):
            imgs = data.train_batch(batch_size=args.batch_size)

            for img in imgs:
                cv2.imshow('Img', img)
                if cv2.waitKey(0) & 0xFF == 27:
                    sys.exit('[*] Esc clicked!')


if __name__ == '__main__':
    main()
