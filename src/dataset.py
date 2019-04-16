import os
import numpy as np

from utils import all_files_under, load_data

class Dataset(object):
    def __init__(self, dataset_name, num_cross_vals, idx_cross_val):
        filenames = all_files_under(os.path.join('../../Data', dataset_name, 'post'), extension='png')

        blocks = []
        num_each_split = int(np.floor(len(filenames) / num_cross_vals))
        for idx in range(num_cross_vals):
            blocks.append(filenames[idx*num_each_split:(idx+1)*num_each_split])

        self.test_data = blocks[idx_cross_val]
        self.val_data = blocks[np.mod(idx_cross_val + 1, num_cross_vals)]
        del blocks[idx_cross_val], blocks[0 if idx_cross_val == len(blocks) else idx_cross_val]
        self.train_data = [item for sub_block in blocks for item in sub_block] + \
                          filenames[-np.mod(len(filenames), num_cross_vals):]

        self.num_train = len(self.train_data)
        self.num_val = len(self.val_data)
        self.num_test = len(self.test_data)

    def train_batch(self, batch_size):
        # batch_files = np.random.choice(self.train_data, batch_size, replace=False)
        batch_files = self.train_data[:batch_size]
        batch_x, batch_y, batch_mask = load_data(batch_files, is_test=False)
        return batch_x, batch_y, batch_mask

    def val_batch(self):
        x, y, mask = load_data(self.val_data, is_test=True)
        return x, y, mask


    def test_batch(self):
        x, y, mask = load_data(self.test_data, is_test=True)
        return x, y, mask
