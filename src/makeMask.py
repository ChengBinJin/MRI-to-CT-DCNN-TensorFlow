import os
import cv2
import argparse

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('--data', dest='data', default='../../Data/brain01', help='dataset for making mask')
args = parser.parse_args()

def main():
    print('Hello World!')



if __name__ == '__main__':
    main()
