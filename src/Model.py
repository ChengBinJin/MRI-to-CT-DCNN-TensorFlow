# ---------------------------------------------------------
# Tensorflow DCNN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import logging
import numpy as np
import tensorflow as tf
import _pickle as cpickle
from tensorflow.python.training import moving_averages

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)

class Model:
    def __init__(self, args, name, input_dims, output_dims, log_path=None):
        self.args = args
        self.name = name
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.log_path = log_path
        self.batch_norm_ops = []

        weight_file_path = '../../Models_zoo/caffe_layers_value.pickle'
        self._read_pretrained_weights(weight_file_path)

        self._init_logger()     # init logger
        self._build_net(self.name)
        self._tensorboard()
        self.show_all_variables(is_train=self.args.is_train)

    def _read_pretrained_weights(self, path):
        with open(path, 'rb') as f:
            self.pretrained_weights = cpickle.load(f, encoding='latin1')

    def _init_logger(self):
        if self.args.is_train:
            formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
            # file handler
            file_handler = logging.FileHandler(os.path.join(self.log_path, 'model.log'))
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            # stream handler
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            # add hanlders
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

    def _build_net(self, name):
        with tf.variable_scope(name):
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, *self.input_dims], name='x')
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, *self.output_dims], name='y')
            self.mode = tf.placeholder(dtype=tf.bool, name='train_mode')

            self.mae = tf.placeholder(dtype=tf.float32, name='MAE')
            self.me = tf.placeholder(dtype=tf.float32, name='ME')
            self.mse = tf.placeholder(dtype=tf.float32, name='MSE')
            self.pcc = tf.placeholder(dtype=tf.float32, name='PCC')

            # Encoding part
            # 256 x 256 x 64
            data = tf.concat([self.x, self.x, self.x], axis=3)
            relu1_1 = self.conv_layer_pretrain(data, 'conv1_1', trainable=True)
            relu1_2 = self.conv_layer_pretrain(relu1_1, 'conv1_2', trainable=True)

            # 128 x 128 x 128
            pool1_2 = self.max_pool_2x2(relu1_2, name='max_pool_1')
            relu2_1 = self.conv_layer_pretrain(pool1_2, 'conv2_1', trainable=True)
            relu2_2 = self.conv_layer_pretrain(relu2_1, 'conv2_2', trainable=True)

            # 64 x 64 x 256
            pool2_2 = self.max_pool_2x2(relu2_2, name='max_pool_2')
            relu3_1 = self.conv_layer_pretrain(pool2_2, 'conv3_1', trainable=True)
            relu3_2 = self.conv_layer_pretrain(relu3_1, 'conv3_2', trainable=True)
            relu3_3 = self.conv_layer_pretrain(relu3_2, 'conv3_3', trainable=True)

            # 32 x 32 x 512
            pool3_3 = self.max_pool_2x2(relu3_3, name='max_pool_3')
            relu4_1 = self.conv_layer_pretrain(pool3_3, 'conv4_1', trainable=True)
            relu4_2 = self.conv_layer_pretrain(relu4_1, 'conv4_2', trainable=True)
            relu4_3 = self.conv_layer_pretrain(relu4_2, 'conv4_3', trainable=True)

            # 16 x 16 x 512
            pool4_3 = self.max_pool_2x2(relu4_3, name='max_pool_4')
            relu5_1 = self.conv_layer_pretrain(pool4_3, 'conv5_1', trainable=True)
            relu5_2 = self.conv_layer_pretrain(relu5_1, 'conv5_2', trainable=True)
            relu5_3 = self.conv_layer_pretrain(relu5_2, 'conv5_3', trainable=True)

            # Decoding
            # 16 x 16 x 512
            relu5_4 = self.conv_layer(relu5_3, output_dim=512, name='conv5_4')
            relu5_5 = self.conv_layer(relu5_4, output_dim=512, name='conv5_5')
            relu5_6 = self.conv_layer(relu5_5, output_dim=512, name='conv5_6')

            # 32 x 32 x 512
            unpool4_3 = self.unpooling2d(relu5_6, name='unpooling_1')
            concat4_3 = tf.concat([relu4_3, unpool4_3], axis=3, name='concat_1')  # 32 x 32 x 1024
            relu4_4 = self.conv_layer(concat4_3, output_dim=512, name='conv4_4')
            relu4_5 = self.conv_layer(relu4_4, output_dim=512, name='conv4_5')
            relu4_6 = self.conv_layer(relu4_5, output_dim=512, name='conv4_6')

            # 64 x 64 x 256
            unpool3_3 = self.unpooling2d(relu4_6, name='unpooling_2')
            concat3_3 = tf.concat([relu3_3, unpool3_3], axis=3, name='concat_2')
            relu3_4 = self.conv_layer(concat3_3, output_dim=256, name='conv3_4')
            relu3_5 = self.conv_layer(relu3_4, output_dim=256, name='conv3_5')
            relu3_6 = self.conv_layer(relu3_5, output_dim=256, name='conv3_6')

            # 128 x 128 x 128
            unpool2_2 = self.unpooling2d(relu3_6, name='unpooling_3')
            concat2_2 = tf.concat([relu2_2, unpool2_2], axis=3, name='concat_3')
            relu2_3 = self.conv_layer(concat2_2, output_dim=128, name='conv2_3')
            relu2_4 = self.conv_layer(relu2_3, output_dim=128, name='conv2_4')

            # 256 x 256 x 64
            unpool1_2 = self.unpooling2d(relu2_4, name='unpooling_4')
            concat1_2 = tf.concat([relu1_2, unpool1_2], axis=3, name='concat_4')
            relu1_3 = self.conv_layer(concat1_2, output_dim=64, name='conv1_3')
            relu1_4 = self.conv_layer(relu1_3, output_dim=64, name='conv1_4')

            # 256 x 256 x 1
            self.pred = self.last_conv2d(relu1_4, output_dim=1, k_h=1, k_w=1, d_h=1, d_w=1, name='conv1_5')

            self.data_loss = self.regress_loss(self.pred, self.y)
            self.reg_term = self.args.weight_decay * tf.reduce_sum(
                [tf.nn.l2_loss(weight) for weight in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            # self.reg_term = self.args.weight_decay * tf.losses.get_regularization_loss(scope=self.name)
            self.total_loss = self.data_loss + self.reg_term

            optim_op = tf.train.AdamOptimizer(self.args.learning_rate).minimize(self.total_loss)
            train_ops = [optim_op] + self.batch_norm_ops
            self.train_op = tf.group(*train_ops)

    def _tensorboard(self):
        tf.summary.scalar('Loss/Total Loss', self.total_loss)
        tf.summary.scalar('Loss/Data Loss', self.data_loss)
        tf.summary.scalar('Loss/Reg Term', self.reg_term)
        self.summary_op = tf.summary.merge_all()

        self.summary_val = tf.summary.merge(inputs=[tf.summary.scalar('Acc/MAE', self.mae),
                                                    tf.summary.scalar('Acc/ME', self.me),
                                                    tf.summary.scalar('Acc/RMSE', self.mse),
                                                    tf.summary.scalar('Acc/PCC', self.pcc)], name='Acc')

    @staticmethod
    def regress_loss(pred, y):
        return tf.reduce_mean(tf.abs(pred - y))

    @staticmethod
    def last_conv2d(x, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, padding='SAME', name='conv2d'):
        with tf.variable_scope(name):
            conv_weights = tf.get_variable(name="W",
                                           shape=[k_h, k_w, x.get_shape()[-1], output_dim],
                                           initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv_biases = tf.get_variable(name="b",
                                          shape=[output_dim],
                                          initializer=tf.constant_initializer(0.))

            conv = tf.nn.conv2d(input=x, filter=conv_weights, strides=[1, d_h, d_w, 1], padding=padding)
            bias = tf.nn.bias_add(value=conv, bias=conv_biases)

            return bias

    def conv_layer(self, x, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02, padding='SAME', name='conv2d',
                   is_print=True):
        with tf.variable_scope(name):
            conv_weights = tf.get_variable(name="W",
                                           shape=[k_h, k_w, x.get_shape()[-1], output_dim],
                                           initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv_biases = tf.get_variable(name="b",
                                          shape=[output_dim],
                                          initializer=tf.constant_initializer(0.))

            conv = tf.nn.conv2d(input=x, filter=conv_weights, strides=[1, d_h, d_w, 1], padding=padding)
            bias = tf.nn.bias_add(value=conv, bias=conv_biases)
            norm = self.batch_norm(bias, name='batch_norm', _ops=self.batch_norm_ops, is_train=self.mode)
            relu = tf.nn.relu(norm)

            if is_print:
                self.print_activations(relu)

        return relu

    def conv_layer_pretrain(self, x, name, trainable=False, is_print=True):
        with tf.variable_scope(name):
            w = self.get_conv_weight(name)
            b = self.get_bias(name)
            conv_weights = tf.get_variable(name="W",
                                           shape=w.shape,
                                           initializer=tf.constant_initializer(w),
                                           trainable=trainable)
            conv_biases = tf.get_variable(name="b",
                                          shape=b.shape,
                                          initializer=tf.constant_initializer(b),
                                          trainable=trainable)

            conv = tf.nn.conv2d(input=x, filter=conv_weights, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(value=conv, bias=conv_biases)
            norm = self.batch_norm(bias, name='batch_norm', _ops=self.batch_norm_ops, is_train=self.mode)
            relu = tf.nn.relu(norm)

            if is_print:
                self.print_activations(relu)

        return relu

    def get_conv_weight(self, name):
        f = self.get_weight(name)
        return f.transpose((2, 3, 1, 0))

    def get_weight(self, layer_name):
        layer = self.pretrained_weights[layer_name]
        return layer[0]

    def get_bias(self, layer_name):
        layer = self.pretrained_weights[layer_name]
        return layer[1]

    @staticmethod
    def batch_norm(x, name, _ops, is_train=True):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable('beta', params_shape, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', params_shape, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))

            if is_train is True:
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                                              initializer=tf.constant_initializer(0.0, tf.float32),
                                              trainable=False)
                moving_variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                                                  initializer=tf.constant_initializer(1.0, tf.float32),
                                                  trainable=False)

                _ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
                _ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                                       initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
                variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                                           initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)

            # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-5)
            y.set_shape(x.get_shape())

            return y

    @staticmethod
    def max_pool_2x2(x, name='max_pool'):
        return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    @staticmethod
    def unpooling2d(x, size=(2, 2), name='unpooling2d'):
        with tf.name_scope(name):
            shape = x.get_shape().as_list()
            output = tf.image.resize_nearest_neighbor(x, size=(size[0] * shape[1], size[1] * shape[2]))
        return output

    @staticmethod
    def print_activations(t):
        logger.info(t.op.name + '{}'.format(t.get_shape().as_list()))

    @staticmethod
    def show_all_variables(is_train=True):
        total_count = 0
        for idx, op in enumerate(tf.trainable_variables()):
            shape = op.get_shape()
            count = np.prod(shape)
            if is_train:
                logger.info("[%2d] %s %s = %s" % (idx, op.name, shape, count))
            else:
                print("[%2d] %s %s = %s" % (idx, op.name, shape, count))

            total_count += int(count)

        if is_train:
            logger.info("[Total] variable size: %s" % "{:,}".format(total_count))
        else:
            print("[Total] variable size: %s" % "{:,}".format(total_count))
