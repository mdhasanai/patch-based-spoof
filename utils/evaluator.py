import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd

def calculate_fpr(y_true, y_pred):
    true_spoof = 0  #### Spoof being 1
    false_real = 0  #### real being 0
    for i in range(len(y_true)):
        target = y_true[i]
        pred = y_pred[i]
        if target:
            true_spoof += 1
            if not pred:
                false_real += 1
    return false_real / true_spoof if true_spoof else 0

def calculate_fnr(y_true, y_pred):
    true_real = 0  #### Spoof being 1
    false_spoof = 0  #### real being 0
    for i in range(len(y_true)):   
        target = y_true[i]
        pred = y_pred[i]
        if not target:
            true_real += 1
            if pred:
                false_spoof += 1
    return false_spoof / true_real if true_real else 0

def calculate_eer(APCER, BPCER):
    return abs(APCER - BPCER)

def get_matric(y_true, y_hat):
    acc   = accuracy_score(y_true, y_hat) * 100
    apcer = calculate_fpr(y_true, y_hat) * 100
    bpcer = calculate_fnr(y_true, y_hat) * 100
    return acc, apcer, bpcer

