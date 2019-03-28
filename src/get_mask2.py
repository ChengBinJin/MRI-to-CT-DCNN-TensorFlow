import argparse
import numpy as np

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--data', dest='data', default='../../Data/brain05/npz_data', help='defaul npz data path')
args = parser.parse_args()

from utils import all_files_under

def main(path):
    print('Hello get_mask2.py!')

    data_paths = all_files_under(path, extension='npz')
    for idx, data_path in enumerate(data_paths):
        print('idx: {}, path: {}'.format(idx, data_path))

        data = np.load(data_path)
        # print('data shape: {}'.format(data['arr_0.npy'].shape))
        # print(data)

if __name__ == '__main__':
    main(args.data)