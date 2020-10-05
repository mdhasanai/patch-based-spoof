import sys
import os.path
sys.path.append('..')
import argparse
import copy
import glob
import math
import time
import cv2
import numpy as np
from trainers.patch_trainer import Trainer
import torch
from torch.nn import functional as F
from models.patch_based_cnn.model import PatchModel
from dataloaders.patch_dataloader import PatchDataset
from torchvision import transforms
import sys
import pandas as pd
from utils.evaluator import get_matric
# sys.setrecursionlimit(1500)


def load_model(im_size, gpus, channel_size=3, freeze_layer=False, resume_training=False):
    model = PatchModel(im_size, channel_size)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model, device_ids=gpus)
    if resume_training:
        model.load_state_dict(torch.load(f"../ckpts/patch_based_cnn/{im_size}/msu_model_new.pth"))
        for param in model.parameters():
            param.requires_grad = True

    if freeze_layer:
        for param in model.parameters():
            param.requires_grad = False

    model.cuda(gpus[0])
    return model

def get_arguments():
    
    parser = argparse.ArgumentParser(description='Define Training Details...')
    parser.add_argument('--im_size', type=int, default=96)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--batch_size', type=int, default=7)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--no_workers', type=int, default=6)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--epoch', type=int, default=8)
    parser.add_argument('--no_patch', type=int, default=15)
    parser.add_argument('--channel_size', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='msu')
    parser.add_argument('--protocol', type=str, default='2')
    parser.add_argument('--protocol_type', type=str, default='1')
    parser.add_argument('--output_dir', type=str, default='../ckpts')
    parser.add_argument('--csv_dir', type=str, default='../log')
    parser.add_argument('--save_epoch_freq', type=int, default=2, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--not_improved', type=int, default=10, help='break if consequtive not improve')
    args = parser.parse_args()
    
    return args

def oulu(protocol=2, proto_type=1):
    csv_root_path = f"/home/ec2-user/SageMaker/hasan/access_paper/gaze-research/csvs/swi_protocol/protocol_{protocol}/protocol_{protocol}_type{proto_type}"
    
    TRAIN_CSV = pd.read_csv(f"{csv_root_path}_train.csv",low_memory=False)
    TEST_CSV  = pd.read_csv(f"{csv_root_path}_test.csv",low_memory=False)
    DEV_CSV   = pd.read_csv(f"{csv_root_path}_dev.csv",low_memory=False) 
    
    train_imgs = glob.glob(f"{msu_train_path}/Train/**/*")
    
    return {"train": train_imgs, "val": val_imgs}

def msu(protocol=2, proto_type=1):
    csv_root_path = f"/home/ec2-user/SageMaker/hasan/access_paper/gaze-research/csvs/siw_protocol/protocol_{protocol}/protocol_{protocol}"
    if protocol>1 and proto_type is not None:
        csv_root_path = f"{csv_root_path}_type{proto_type}"
    if protocol ==1 and proto_type is not None:
        print(f"proto_type should be None for protocol 1")
    elif protocol>1 and proto_type is None:
        print(f"proto_type should not be none for protocol {protocol}")
   
    TRAIN_CSV = pd.read_csv(f"{csv_root_path}_train.csv",low_memory=False)
    TEST_CSV  = pd.read_csv(f"{csv_root_path}_test.csv",low_memory=False)
    DEV_CSV   = pd.read_csv(f"{csv_root_path}_dev.csv",low_memory=False) 
    print("DATASET INFORMATION: ")
    print("MSU_SIW Protocol: ",protocol)
    if proto_type is not None:
        print("MSU_SIW Protocol Type: ",proto_type)
    print(f"TRAIN IMAGES: {len(TRAIN_CSV)} \nTEST IMAGES: {len(TEST_CSV)} \nDEV IMAGES: {len(DEV_CSV)}")
    return {"train": TRAIN_CSV, "val": DEV_CSV,"test":TEST_CSV}

def TensorToImage(imgs):
    """
    imgs: should be (3,h,w) of tensor
    """
    import matplotlib.pyplot as plt
    plt.imshow( imgs.permute(1, 2, 0)  )

def mobile_replay(proto=None):
    mobile_train_path = "/home/ec2-user/SageMaker/dataset/spoof-data/mobile_replay"
    train_imgs = glob.glob(f"{mobile_train_path}/train/***/**/*")
    val_imgs = glob.glob(f"{mobile_train_path}/test/***/**/*")
    return {"train": train_imgs, "val": val_imgs}



def main(): 

    args       = get_arguments()
    im_size    = args.im_size
    epochs     = args.epoch
    iterations = 6 - int(math.log((im_size // 3), 2))
    no_patch = int(args.no_patch)
    batch_size = args.batch_size
    color_mode = ['rgb']
    channel_size = int(args.channel_size)
    total_small_patches = sum([4 ** i for i in range(iterations)]) if no_patch is None else no_patch
    os.makedirs(f"{args.output_dir}/patch_based_cnn/{args.im_size}/",exist_ok=True)
    print(f"Number of small patches for {im_size} is {total_small_patches}.")
    # getting data
    get_data = globals().get(args.dataset)
    im_data  = get_data()
    oulu
    transform = transforms.Compose([transforms.ToTensor()])
    dataloaders = {}
    for phase in ['train', 'val']:
        print(len(im_data[phase]))
        dset = PatchDataset(csv_data=im_data[phase], transform=transform, im_size=im_size, color_mode=color_mode, no_patch=no_patch, phase=phase)
        dataloaders[phase] = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=True, num_workers=7)

    gpus = [int(x.strip()) for x in args.gpu.split(",")]
    print(gpus)
    model = load_model(im_size, gpus, channel_size=channel_size)
    loss = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())
    Trainer(gpus[0],args, get_matric).train_model(model, loss, dataloaders, optimizer, epochs)

    
main()