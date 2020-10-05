import numpy as np
import torch
import os
import glob
import pandas as pd
import argparse


datatype = "oulu"
taw_list = np.arange(0.10, 1.00, .01).tolist()
beta = 0.5
parser = argparse.ArgumentParser(description='PASS NECESSARY ARGS')
parser.add_argument('--dev', help='path to dev.csv is required', required=True)
parser.add_argument('--test', help='path to test.csv is required', required=True)
args = vars(parser.parse_args())

dev_path = args['dev']
test_path = args['test']
df_dev = pd.read_csv(dev_path, names=['frame', 'score'])
df_test = pd.read_csv(test_path, names=['frame', 'score'])
dict_taw = {}
print(df_dev.head())

wrong_img_false_real_dev = []
wrong_img_false_spoof_dev = []
correct_img_false_real_dev = []
correct_img_false_spoof_dev = []

wrong_img_false_real_test = []
wrong_img_false_spoof_test = []
correct_img_false_real_test = []
correct_img_false_spoof_test = []


def get_labels(row, taw, datatype="msu"):
    pred = 0 if float(row['score']) < taw else 1
    if datatype == 'msu':
        label = row['frame'].split("/")[-2]
        target = 0 if label == "live" else 1
    elif datatype == "oulu":
        label = os.path.split(row['frame'])[-1].split(".")[0].split("_")[-1]
        target = 0 if label == "1" else 1
    return pred, target



def create_data(df, taw):
    data = []
    for i, row in df.iterrows():
        pred, target = get_labels(row, taw, datatype=datatype)
        data.append([pred, target, row['score'], row['frame']])
        
    return data

def fetch_single_row(row):
    pred = row[0]
    target = row[1]
    score = row[2]
    frame = row[3]
    return pred, target, score, frame

def calculate_fpr(false_real_lis, data):
    true_spoof = 0  #### Spoof being 1
    false_real = 0  #### real being 0
    for i in range(len(data)):

        pred, target, score, frame = fetch_single_row(data[i])
        if target:
            true_spoof += 1
            if not pred:
                false_real += 1
                d = [frame, score]
                if float(score) < .5 and d not in false_real_lis:
                    false_real_lis.append(d)
    return false_real / true_spoof, false_real_lis


def calculate_fnr(false_spoof_lis, data):
    true_real = 0  #### Spoof being 1
    false_spoof = 0  #### real being 0
    for i in range(len(data)):   
        pred, target, score, frame = fetch_single_row(data[i])
        if not target:
            true_real += 1
            if pred:
                false_spoof += 1
                d = [frame, score]
                if float(score) > .5 and d not in false_spoof_lis:
                    false_spoof_lis.append(d)
    return false_spoof / true_real, false_spoof_lis 


def calculate_wer(APCER, BPCER, beta):
    return beta * APCER + (1 - beta) * BPCER


def calculate_hter(APCER, BPCER):
    return (APCER + BPCER) / 2

def calculate_eer(APCER, BPCER):
    return abs(APCER - BPCER)

def find_optimal_taw(dev_score_df, taw_range, taw_record):
    global wrong_img_false_real_dev
    global wrong_img_false_spoof_dev
    beta = 0.5
    for t in taw_range:
        data = create_data(dev_score_df, t)
        APCER, wrong_img_false_real_dev = calculate_fpr(wrong_img_false_real_dev, data)
        BPCER, wrong_img_false_spoof_dev = calculate_fnr(wrong_img_false_spoof_dev, data)
        print(f'taw: {t:.2f}, APCER: {APCER:.7f}, BPCER: {BPCER:.7f}')
        if t not in taw_record.keys():
            taw_record[t] = calculate_eer(APCER, BPCER)
    print(f"wrong detections: Spoof but said real: ")
#     print(wrong_img_false_real_dev)
    
    print(f"wrong detections: Real but said spoof: ")
#     print(wrong_img_false_spoof_dev)
    return min(taw_record, key=taw_record.get)


# calculated WER on test set, calculate HTER on test set on optimal threshold
optimal_threshold = find_optimal_taw(df_dev, taw_list, dict_taw)
data = create_data(df_test, optimal_threshold)
test_apcer, wrong_img_false_real_test = calculate_fpr(wrong_img_false_real_test, data)
test_bpcer,  wrong_img_false_spoof_test = calculate_fnr(wrong_img_false_spoof_test, data)

print(f"wrong detections: Spoof but said real: ")
# print(wrong_img_false_real_test)
    
print(f"wrong detections: Real but said spoof: ")
# print(wrong_img_false_spoof_test)

print(f'OPTIMAL THRESHODL: {optimal_threshold}')
print(f'WER: {calculate_wer(test_apcer, test_bpcer, beta)}\nHTER: {calculate_hter(test_apcer, test_bpcer)}')
print(f'APCER: {test_apcer}\nBPCER: {test_bpcer}')
print(f'ACER: {(test_apcer+test_bpcer)/2}')