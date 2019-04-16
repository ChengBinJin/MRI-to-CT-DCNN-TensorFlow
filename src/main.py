# ---------------------------------------------------------
# Tensorflow DCNN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

from dataset import Dataset
from model import Model
from solver import Solver

parser = argparse.ArgumentParser(description='main')
parser.add_argument('--gpu_index', dest='gpu_index', default='0',
                    help='gpu index if you have multiple gpus, default: 0')
parser.add_argument('--is_train', dest='is_train', default=False, action='store_true',
                    help='train mode, default: False')
parser.add_argument('--batch_size', dest='batch_size', default=4, type=int,
                    help='batch size for one iteration')
parser.add_argument('--dataset', dest='dataset', default='brain01',
                    help='dataset name, default: brain01')
parser.add_argument('--learning_rate', dest='learning_rate', default=1e-3, type=float,
                    help='learning rate, default: 2e-4')
parser.add_argument('--weight_decay', dest='weight_decay', default=1e-4, type=float,
                    help='weight decay, default: 1e-5')
parser.add_argument('--epoch', dest='epoch', default=600, type=int,
                    help='number of epochs, default: 600')
parser.add_argument('--print_freq', dest='print_freq', default=10, type=int,
                    help='print frequency for loss, default: 100')
parser.add_argument('--load_model', dest='load_model', default=None,
                    help='folder of saved model that you wish to continue training, '
                         '(e.g., 20190411-2217), default: None')
args = parser.parse_args()

def make_folders(is_train=True, load_model=None, dataset=None):
    model_dir, log_dir, sample_dir, test_dir = None, None, None, None

    if is_train:
        if load_model is None:
            cur_time = datetime.now().strftime("%Y%m%d-%H%M")
            model_dir = "model/{}/{}".format(dataset, cur_time)

            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
        else:
            cur_time = load_model
            model_dir = "model/{}/{}".format(dataset, cur_time)

        sample_dir = "sample/{}/{}".format(dataset, cur_time)
        log_dir = "logs/{}/{}".format(dataset, cur_time)

        if not os.path.isdir(sample_dir):
            os.makedirs(sample_dir)

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

    else:
        model_dir = "model/{}/{}".format(dataset, load_model)
        log_dir = "logs/{}/{}".format(dataset, load_model)
        test_dir = "test/{}/{}".format(dataset, load_model)

        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

    return model_dir, sample_dir, log_dir, test_dir


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index

    model_dir, sampel_dir, log_dir, test_dir = make_folders(is_train=args.is_train,
                                                            load_model=args.load_model,
                                                            dataset=args.dataset)

    # Initialize session
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    sess = tf.Session(config=run_config)

    # Initialize model and solver
    num_cross_vals = 6  # num_cross_vals have to bigger than 3 (train dataset, validation dataset, and test dataset)
    model = Model(args, name='UNet', input_dims=(256, 256, 1), output_dims=(256, 256, 1), log_path=log_dir)
    solver = Solver(sess, model)

    for idx_cross_val in range(num_cross_vals):
        print('idx_cross_val: {}'.format(idx_cross_val))

        data = Dataset(args.dataset, num_cross_vals, 5)
        sess.run(tf.global_variables_initializer())

        num_iters = int(args.epoch * data.num_train / args.batch_size)
        for iter_time in range(num_iters):
            mrImgs, ctImgs, maskImgs = data.train_batch(batch_size=args.batch_size)
            _, total_loss, data_loss, reg_term, mrImgs_, preds, ctImgs_ = solver.train(mrImgs, ctImgs, maskImgs)

            if np.mod(iter_time, args.print_freq) == 0:
                print('{} / {} Total Loss: {}, Data Loss: {}, Reg Term: {}'.format(
                    iter_time, num_iters, total_loss, data_loss, reg_term))

            if np.mod(iter_time, 100) == 0:
                solver.save_imgs(mrImgs_, ctImgs_, preds, iter_time, save_folder=sampel_dir)

            # if np.mod(iter_time, 1000) == 0:
            #     print('Evaluate...')
            #     mrImgs, ctImgs, maskImgs = data.val_batch()
            #     preds = solver.evaluate(mrImgs, ctImgs, maskImgs, batch_size=args.batch_size)
            #
            #     solver.save_imgs(mrImgs, ctImgs, preds, iter_time, save_folder=sampel_dir)





if __name__ == '__main__':
    main()
