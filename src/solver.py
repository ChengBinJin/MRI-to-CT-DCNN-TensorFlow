import os
import cv2
import numpy as np

from utils import intransform

class Solver(object):
    def __init__(self, sess, model):
        self.sess = sess
        self.model = model

        # self.dataset = Dataset(self.flags.dataset)

    def train(self, x, y, mask):
        feed = {
            self.model.x: x,
            self.model.y: y,
            self.model.mode: True
        }

        train_op = self.model.train_op
        total_loss = self.model.total_loss
        data_loss = self.model.data_loss
        reg_term = self.model.reg_term

        mrImgs = self.model.x
        preds = self.model.pred
        ctImgs = self.model.y

        return self.sess.run([train_op, total_loss, data_loss, reg_term, mrImgs, preds, ctImgs], feed_dict=feed)

    def evaluate(self, x, y, mask, batch_size):
        num_data = x.shape[0]
        preds = np.zeros_like(x)

        for i in range(0, num_data, batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            # mask_batch = mask[i:i+batch_size]

            feed = {
                self.model.x: x_batch,
                self.model.y: y_batch,
                self.model.mode: False
            }

            step_preds = self.sess.run(self.model.pred, feed_dict=feed)
            preds[i:i+batch_size] = step_preds

        return preds

    @staticmethod
    def save_imgs(x, y, pred, id_, save_folder=None):
        num_data, h, w, c = x.shape

        for i in range(num_data):
            canvas = np.zeros((h, 3*w), dtype=np.uint8)
            canvas[:, :w] = intransform(x[i])           # Input MR image
            canvas[:, w:2*w] = intransform(pred[i])     # Predicted CT image
            canvas[:, -w:] = intransform(y[i])          # GT CT image

            imgName = os.path.join(save_folder, '{}_{}'.format(str(id_).zfill(6), str(i).zfill(3))) + '.png'
            cv2.imwrite(imgName, canvas)

