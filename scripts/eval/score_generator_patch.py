import numpy as np
import torch
import os
import glob
import pandas as pd
import argparse


# datatype = "oulu"
taw_list = np.arange(0.10, 1.00, .01).tolist()
beta = 0.5
parser = argparse.ArgumentParser(description='PASS NECESSARY ARGS')
parser.add_argument('--train_dataset', help='path to dev.csv is required', default="oulu")
# parser.add_argument('--test', help='path to test.csv is required', required=True)
args = vars(parser.parse_args())

# dev_path = args['dev']
# test_path = args['test']


wrong_img_false_real_dev = []
wrong_img_false_spoof_dev = []
correct_img_false_real_dev = []
correct_img_false_spoof_dev = []

wrong_img_false_real_test = []
wrong_img_false_spoof_test = []
correct_img_false_real_test = []
correct_img_false_spoof_test = []


def get_labels(row, datatype="oulu"):
#     pred = 0 if float(row['score']) < taw else 1
#     print(datatype)
    if datatype == 'msu':
        label = row['frame'].split("/")[-2]
#         print(row['frame'])
        target = 0 if (label == "real" or label == "live") else 1
    
    elif datatype == "oulu":
        label = row['frame'].split("/")
        if "spoof" in label or "device" in label or "print" in label or "print2" in label or "video-replay2" in label:
            target = 1
        elif "print1" in label or "video-replay1" in label:
            target = 1
        elif "live" in label or "real" in label:
            target = 0
    elif datatype == "mobile_replay":
        label = row['frame'].split("/")[-3]
        target = 1 if label == "spoof" else 0

    return target



def create_data(df, dset):
    data = []
    for i, row in df.iterrows():
        target = get_labels(row, datatype=dset)
        data.append([target, float(row['score']), row['frame']])
        
    return data

def fetch_single_row(row):
#     pred = row[0]
    target = row[0]
    score = row[1]
    frame = row[2]
    return target, score, frame

def calculate_fpr(data, taw):
    true_spoof = 0  #### Spoof being 1
    false_real = 0  #### real being 0
    for i in range(len(data)):

        target, score, frame = fetch_single_row(data[i])
        pred = 0 if score < taw else 1
        if target:
            true_spoof += 1
            if not pred:
                false_real += 1
#                 d = [frame, score]
#                 if float(score) < .5 and d not in false_real_lis:
#                     false_real_lis.append(d)
    return false_real / true_spoof if true_spoof else 0


def calculate_fnr(data, taw):
    true_real = 0  #### Spoof being 1
    false_spoof = 0  #### real being 0
#     print(len(data))
    for i in range(len(data)):   
        target, score, frame = fetch_single_row(data[i])
        pred = 0 if score < taw else 1
#         print(target, pred)
        if not target:
            true_real += 1
            if pred:
                false_spoof += 1
#                 d = [frame, score]
#                 if float(score) > .5 and d not in false_spoof_lis:
#                     false_spoof_lis.append(d)
    return false_spoof / true_real if true_real else 0


def calculate_wer(APCER, BPCER, beta):
    return beta * APCER + (1 - beta) * BPCER


def calculate_hter(APCER, BPCER):
    return (APCER + BPCER) / 2

def calculate_eer(APCER, BPCER):
    return abs(APCER - BPCER)

def find_optimal_taw(dev_score_df, taw_range, taw_record, dset):
 
    beta = 0.5
    data = create_data(dev_score_df, dset)
    for t in taw_range:
        APCER = calculate_fpr(data, t)
        BPCER = calculate_fnr(data, t)
        print(f'taw: {t:.2f}, APCER: {APCER:.7f}, BPCER: {BPCER:.7f}')
        if t not in taw_record.keys():
            taw_record[t] = calculate_eer(APCER, BPCER)
#     print(f"wrong detections: Spoof but said real: ")
#     print(wrong_img_false_real_dev)
    
#     print(f"wrong detections: Real but said spoof: ")
#     print(wrong_img_false_spoof_dev)
    return min(taw_record, key=taw_record.get)
init_path = 'score_data/patch'
train_dataset = args["train_dataset"]
#eval_datasets = ["msu", "oulu", "mobile_replay"]
eval_datasets = ["oulu"]
im_sizes = [48]#[12, 24, 48, 96]

print(f"Working on {train_dataset} trained model.")
for im_size in im_sizes:
    for dset in eval_datasets:
        print(f"Inferring for {im_size} sized images on {dset} dataset.")
        
        dev_path = f"{init_path}/{train_dataset}/val/{dset}_{im_size}.csv"
        test_path = f"{init_path}/{train_dataset}/test/{dset}_{im_size}.csv"
        df_dev = pd.read_csv(dev_path, names=['frame', 'score'])
        df_test = pd.read_csv(test_path, names=['frame', 'score'])
        dict_taw = {}
#         print(df_dev.head())
        # calculated WER on test set, calculate HTER on test set on optimal threshold
        optimal_threshold = find_optimal_taw(df_dev, taw_list, dict_taw, dset)
        data = create_data(df_test, dset)
        test_apcer = calculate_fpr(data, optimal_threshold)
        test_bpcer = calculate_fnr(data, optimal_threshold)

        # print(f"wrong detections: Spoof but said real: ")
        # print(wrong_img_false_real_test)

        # print(f"wrong detections: Real but said spoof: ")
        # print(wrong_img_false_spoof_test)
        
        print(f'OPTIMAL THRESHODL: {optimal_threshold}')
        print(f'WER: {calculate_wer(test_apcer, test_bpcer, beta)}\nHTER: {calculate_hter(test_apcer, test_bpcer)}')
        print(f'APCER: {test_apcer}\nBPCER: {test_bpcer}')
        print(f'ACER: {(test_apcer+test_bpcer)/2}')
        if not os.path.exists(f"{init_path}/{train_dataset}/results"):
            os.makedirs(f"{init_path}/{train_dataset}/results")
        f = open(f"{init_path}/{train_dataset}/results/{dset}.txt","a+")
        f.write(f'IMAGE SIZE: {im_size}\n\n')
        f.write(f'OPTIMAL THRESHODL: {optimal_threshold}\n')
        f.write(f'WER: {calculate_wer(test_apcer, test_bpcer, beta)*100}\nHTER: {calculate_hter(test_apcer, test_bpcer)*100}\n')
        f.write(f'APCER: {test_apcer*100}\nBPCER: {test_bpcer*100}\n')
        f.write(f'ACER: {((test_apcer+test_bpcer)/2)*100}\n\n\n')
        f.close()