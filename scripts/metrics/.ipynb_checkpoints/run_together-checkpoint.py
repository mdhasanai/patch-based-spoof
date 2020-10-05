import os
import argparse

parser = argparse.ArgumentParser(description='PASS NECESSARY ARGS')
parser.add_argument('--train_dataset', help='path to dev.csv is required', default="msu")
args = vars(parser.parse_args())

os.system('python dataframe_generator.py --train_dataset mobile_replay')
print("Dataframes are generated!")
os.system('python score_generator_patch.py --train_dataset mobile_replay')
print("Scores are generated!")