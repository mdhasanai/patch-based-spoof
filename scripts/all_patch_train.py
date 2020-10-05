import os
import argparse

parser = argparse.ArgumentParser(description='PASS NECESSARY ARGS')
parser.add_argument('--train_dataset', help='path to dev.csv is required', default="msu")
args = vars(parser.parse_args())

# print("MSU Training Starter for 48 patch")
# os.system('python patch_train.py --im_size 48 --patch_size 15 --dataset msu')

# print("OULU Training Starter for 48 patch")
# os.system('python patch_train.py --im_size 48 --patch_size 15 --dataset oulu --oulu_protocol Protocol_1')

# print("OULU Training Starter for 48 patch")
# os.system('python patch_train.py --im_size 48 --patch_size 15 --dataset oulu --oulu_protocol Protocol_2')

print("MSU Training Starter for 96 patch")
os.system('python patch_train.py --im_size 96 --no_patch 15 --dataset msu --protocol 2 protocol_type 1')

# print("OULU Training Starter for 48 patch")
# os.system('python patch_train.py --im_size 48 --patch_size 15 --dataset oulu --oulu_protocol Protocol_4')

