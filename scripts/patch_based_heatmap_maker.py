import sys
sys.path.append('..')
import glob
import argparse
import numpy as np
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt
import random
import matplotlib
import json
# from patch_based_cnn_trainer import PatchModel
import torch, cv2, h5py, math
import torch.nn.functional as F
from torchvision import transforms
import cv2, os, glob, time
from models.patch_based_cnn.model import PatchModel
from dataloaders.PatchTopDataset import PatchTopDataset
from tqdm import tqdm
import pandas as pd

def load_model(args, im_size, gpus, channel_size=3, freeze_layer=True, resume_training=False):
    model = PatchModel(im_size, channel_size)
    try:
        if torch.cuda.device_count() >= 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model, device_ids=gpus)
        if int(args.resume_training)==1:
            print("Loading Checkpoint..")
            print(f"{args.output_dir}/patch_based_cnn/{args.dataset}/{args.protocol}/{args.protocol_type}/{im_size}/BEST.pth")
            model.load_state_dict(torch.load(f"{args.output_dir}/patch_based_cnn/{args.dataset}/{args.protocol}/{args.protocol_type}/{im_size}/BEST.pth"))
    except:
        model = PatchModel(im_size, channel_size)
        
        if int(args.resume_training)==1:
            print("Loading Checkpoint..")
            print(f"{args.output_dir}/patch_based_cnn/{args.dataset}/{args.protocol}/{args.protocol_type}/{im_size}/BEST.pth")
            model.load_state_dict(torch.load(f"{args.output_dir}/patch_based_cnn/{args.dataset}/{args.protocol}/{args.protocol_type}/{im_size}/BEST.pth"))

    if freeze_layer:
        for param in model.parameters():
            param.requires_grad = False

    model = model.cuda(gpus[0])
    return model

def get_arguments():
    
    parser = argparse.ArgumentParser(description='Define Training Details...')
    parser.add_argument('--im_size', type=int, default=96)
    parser.add_argument('--resume_training', type=int, default=1)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--no_workers', type=int, default=6)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--no_patch', type=int, default=15)
    parser.add_argument('--channel_size', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='oulu')
    parser.add_argument('--protocol', type=str, default='2')
    parser.add_argument('--protocol_type', type=str, default='1')
    parser.add_argument('--output_dir', type=str, default='../ckpts')
    parser.add_argument('--csv_dir', type=str, default='../log')
    parser.add_argument('--save_epoch_freq', type=int, default=2, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--not_improved', type=int, default=10, help='break if consequtive not improve')
    args = parser.parse_args()
    
    return args

def balance_data_for_patch(DF,size=200, dtype="oulu"):
    print("Balancing")
    # separating Real
    if dtype=="oulu":
        REAL  = DF.loc[(DF['label'] == 1)]
        SPOOF = DF.loc[(DF['label'] == 0)]
    else:
        REAL  = DF.loc[(DF['label'] == 0)]
        SPOOF = DF.loc[(DF['label'] == 1)]
    #shuffel
    REAL = REAL.sample(frac=1).reset_index(drop=True)
    REAL = REAL.iloc[0:size]
    REAL = REAL.sort_values(by=['video_id'], ascending=True)
    
    SPOOF = SPOOF.sample(frac=1).reset_index(drop=True)
    SPOOF = SPOOF.iloc[0:size]
    SPOOF = SPOOF.sort_values(by=['video_id'], ascending=True)
    
    print(len(REAL),len(SPOOF))
    return REAL,SPOOF

def oulu(size=200):
    root_path = f"/home/ec2-user/SageMaker/datasets/spoof_datasets/oulu/Dev/crop/"
    REAL  = glob.glob(f"{root_path}/real/**/*")
    SPOOF = glob.glob(f"{root_path}/spoof/**/*")
    random.shuffle(REAL)
    random.shuffle(SPOOF)
    return REAL[:size], SPOOF[:size]

def msu(protocol=2, proto_type=1):
    csv_root_path = f"/home/ec2-user/SageMaker/hasan/access_paper/gaze-research/csvs/siw_protocol/protocol_{protocol}/protocol_{protocol}"
    if protocol>1 and proto_type is not None:
        csv_root_path = f"{csv_root_path}_type{proto_type}"
    if protocol ==1 and proto_type is not None:
        print(f"proto_type should be None for protocol 1")
    elif protocol>1 and proto_type is None:
        print(f"proto_type should not be none for protocol {protocol}")
   
    DEV_CSV   = pd.read_csv(f"{csv_root_path}_dev.csv",low_memory=False) 
    print("DATASET INFORMATION: ")
    print("MSU_SIW Protocol: ",protocol)
    if proto_type is not None:
        print("MSU_SIW Protocol Type: ",proto_type)
    #SPOOF = DEV_CSV.loc[(DEV_CSV['label']==1)]
    #REAL  = 
    print(f"IMAGES: TEST IMAGES: {len(TEST_CSV)} \nDEV IMAGES: {len(DEV_CSV)}")
    return {"train": TRAIN_CSV, "val": DEV_CSV,"test":TEST_CSV}




def get_top_n(accuracy, n):
    top_n_indices = []
    prev = 100
    i = 1
    data = np.unique(-np.sort(-accuracy.flatten()))
    while True:
        top_n_val = data[-i]
#         print(top_n_val)
        if top_n_val == prev:
            conitnue

        lis = np.argwhere(np.logical_and(accuracy >= top_n_val, accuracy<prev))
#         print(lis)
        top_n_indices.extend(lis)
        if len(top_n_indices) >= n:
            break
        prev = top_n_val
        i += 1
#     print(top_n_indices)
    return top_n_indices[:n]


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# def divide_single_img_into_patches(img, size=(224, 224), patch_size=(96, 96, 3), step=1):

#     img = cv2.resize(img, size)
#         #             print(img.shape, len(patch_size))
#     patch_grid = view_as_windows(img, patch_size, step)
#         #         print(patch_grid.shape)
#     return patch_grid

# def get_random_patches(img_patch):
# #     print(img_patch.shape)
#     patch_96x96 = divide_single_img_into_patches(img_patch, size=(96, 96),
#                                                         patch_size=(48, 48, 3)).reshape((-1, 48, 48, 3))
# #     print(patch_96x96.shape)
#     random_20 = np.random.choice(np.arange(patch_96x96.shape[0]), 20)
#     selected_imgs = patch_96x96[random_20]

#     imgs = torch.zeros((20, 3, 48, 48))
#     for idx in range(20):
#         imgs[idx] = transform(selected_imgs[idx])
         

#     return imgs, random_20
def count_correct_patches(dataloader, total_combined, correct_combined, spoof=True, size=(48, 48, 3)):
    start = time.time()
    accurate = 0
    total = 0
    
#     print(dat)
    all_indices = dict.fromkeys(range(0, length * length), 0)
    correct_indices = dict.fromkeys(range(0, length * length), 0)
    cnt = 0
    predictions = []
    running_corrects = 0.0
    model.eval()
    tq = tqdm(dataloader)
    for idx, (im, label) in enumerate(tq):
        #         print(im_path)
#         im = im.permute([0, 2, 3, 1]).numpy().reshape(96, 96, 3)
#         imgs, random_20 = get_random_patches(im)
        
#         print(im.shape)
        imgs = im.reshape(-1, *size)
        label = label.reshape(-1)
        imgs = imgs.cuda(0)
        label = label.type(torch.LongTensor).cuda(0)
#         print(im.size(0))
#         out = model(im)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == label.data)
        total += imgs.size(0)
#         print(running_corrects, total)
#         print(preds, im.size(0))
        for num in range(imgs.size(0)):
#             pred = preds[num]
            if preds[num] == label[num]:
                correct_indices[num] += 1
                correct_combined += 1
            all_indices[num] += 1

            total_combined += 1
            
        acc = (running_corrects/total)*100
        tq.set_postfix(iter=idx, acc=acc)

#         if cnt % 50 == 0:
#             print(f"{cnt+1} images done so far!!!")
        cnt += 1
    
#         print(correct_indices)
#         print('(============================================)')
#         print(all_indices)
    accuracy = (running_corrects/total)*100
    
    if spoof:
        print(f"Spoof test took {time.time() - start} and accuracy is {accuracy}%")
    else:
        print(f"Live test took {time.time() - start} and accuracy is {accuracy}%")

    return correct_indices, accuracy, all_indices, total_combined, correct_combined, preds




def get_patch_accuracy(correct_indices, all_indices):
    patch_wise_accuracy = {key: (correct_indices[key]/all_indices[key]) *
                           100 if all_indices[key] != 0 else 0 for key in range(0, length*length)}
    return np.array([*patch_wise_accuracy.values()]).reshape(length, length)


def plot_heatmap(data, data_type):
    fig, ax = plt.subplots(figsize=(20, 20))

    im, cbar = heatmap(data, np.arange(0, length), np.arange(0, length), ax=ax,
                       cmap="magma", cbarlabel=data_type)

    plt.savefig(f'{data_type}-heatmap.png', bbox_inches='tight', pad_inches=0)
    fig.tight_layout()
    plt.show()







def get_low_n(accuracy, n):
    low_n_indices = []
    prev = 100
    i = 1
    data = np.unique(-np.sort(-accuracy.flatten()))

    while True:
        low_n_val = data[i]

        i += 1
        if low_n_val == prev:
            continue

        lis = np.argwhere(accuracy < low_n_val)
#         print(top_n_val)
        low_n_indices.extend(lis.flatten())
        if len(low_n_indices) >= n:
            break
        prev = low_n_val
        
#     print(top_n_indices)
    return low_n_indices
# com[66, 60]
# low = get_top_n(com.flatten(), 20)

def get_oulu_data(oulu_path):
#     oulu_path = "/media/neuron/spine/Archive/OULU_PROTO"
    protos = os.listdir(oulu_path)
#     train_imgs = []
    val_imgs = []
    for proto in protos:
#         train_imgs.extend(glob.glob(f"{oulu_path}/{proto}/train/**/*"))
        val_imgs.extend(glob.glob(f"{oulu_path}/{proto}/Test/**/*"))

    return val_imgs


def count_correct_patches(dataloader, total_combined, correct_combined, spoof=True, size=(48, 48, 3)):
    start = time.time()
    accurate = 0
    total = 0
    
#     print(dat)
    all_indices = dict.fromkeys(range(0, length * length), 0)
    correct_indices = dict.fromkeys(range(0, length * length), 0)
    cnt = 0
    predictions = []
    running_corrects = 0.0
    model.eval()
    
    tq = tqdm(dataloader)
    for idx, (im, label) in enumerate(tq):

        imgs = im.reshape(-1, *size)
        label = label.reshape(-1)
        imgs = imgs.cuda(0)
        label = label.type(torch.LongTensor).cuda(0)

        outputs = model(imgs)
        probs, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == label.data)
        total += imgs.size(0)
        mean_prob = torch.mean(probs)
        
        for num in range(imgs.size(0)):
#             pred = preds[num]
            if preds[num] == label[num]:
                correct_indices[num] += 1
                correct_combined += 1
            all_indices[num] += 1

            total_combined += 1
        acc = (running_corrects/total)*100
        tq.set_postfix(iter=idx, acc=acc)
        #if cnt % 50 == 0:
            #print(f"{cnt+1} images done so far!!!")
        cnt += 1
    

    accuracy = (running_corrects/total)*100
    
    if spoof:
        print(f"Spoof test took {time.time() - start} and accuracy is {accuracy}%")
    else:
        print(f"Live test took {time.time() - start} and accuracy is {accuracy}%")

    return correct_indices, accuracy, all_indices, total_combined, correct_combined, preds



def run_on_oulu(oulu_data_path, accuracy, number_of_patches=100):
    accuracy = accuracy.flatten()
    top_n_indices = get_top_n(accuracy, number_of_patches)
    number_of_patches = len(top_n_indices)
    print(f"Number of patches: {number_of_patches}")
    start = time.time()
    accurate = 0
    val_data = get_oulu_data(oulu_data_path)
    val_data = list(np.random.choice(val_data, n_samples))
    print(f"Number of data {len(val_data)}")
#     print(dat)
    all_indices = dict.fromkeys(range(0, length * length), 0)
    correct_indices = dict.fromkeys(range(0, length * length), 0)
    cnt = 0
#     predictions = []
    batch = int(2500//number_of_patches)
    img_data = []
    total_batch_corr = [0] * number_of_patches
    total_top_2_batch_corr = [0] * number_of_patches
    total_top_2_batch_corr_with_prior = [0] * number_of_patches
#     total_done = 0
    
    for index, im_path in enumerate(val_data):
        #         print(im_path)
#         predictions.append([])
        img = cv2.imread(im_path)
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_data.append([im_path, img])
        
        if len(img_data) == batch or ((index == len(val_data) - 1)):
            batch = len(img_data)
            print(f"Infering on {index+1} images!")
            inputs = np.zeros((batch*number_of_patches, 96, 96, 3))
            for idx, (im_path, img) in enumerate(img_data):
                inputs[idx*number_of_patches:(idx+1)*number_of_patches] = get_batch_imgs(divide_single_img_into_patches(img, step=step_size).reshape((-1, 96, 96, 3)), top_n_indices)

            x = model.predict_on_batch(inputs)
            preds = np.array(np.argmax(x, axis=1)).reshape(batch, -1)
            probs = np.array(np.max(x, axis=1)).reshape(batch, -1)
#             for k in range(3):    
#                 print(probs[k], preds[k])
            for batch_idx in range(batch):
                batch_corr = []
                top_2_batch_corr = []
                top_2_batch_corr_with_prior = []
#                 total_batch_corr = 0
                spoof = False if img_data[batch_idx][0].split("/")[-2] ==  "real" else True
                
                for num, pred in enumerate(preds[batch_idx]):
                    p = 0 if num == 0 else batch_corr[num-1]
                    p2 = 0 if num == 0 else top_2_batch_corr[0] 
                    
                    batch_corr.append(p)
                    top_2_batch_corr.append(p2)
                    top_2_batch_corr_with_prior.append(p2)
                    if spoof:
                        if pred == 1:
                            correct_indices[top_n_indices[num]] += 1
                            batch_corr[num] += 1
                            top_2_batch_corr[num] += 1
                            if top_2_batch_corr_with_prior[0] != 1 and num > 0:
                                top_2_batch_corr_with_prior[num] += (1 if probs[batch_idx][num] > probs[batch_idx][0] else 0)
                            else:
                                top_2_batch_corr_with_prior[num] += 1
                    else:
                        if pred == 0:
                            correct_indices[top_n_indices[num]] += 1

                            batch_corr[num] += 1
                            top_2_batch_corr[num] += 1 
                            if top_2_batch_corr_with_prior[0] != 1 and num > 0:
                                top_2_batch_corr_with_prior[num] += (1 if probs[batch_idx][num] > probs[batch_idx][0] else 0)
                            else:
                                top_2_batch_corr_with_prior[num] += 1
#                 print(batch_corr)
                for idx, corr in enumerate(batch_corr):
                    thresh = (int((idx+1)/2)) 
                    if corr > thresh:
                        total_batch_corr[idx] += 1
                    if top_2_batch_corr[idx] >= 1:
                        total_top_2_batch_corr[idx] += 1
                    if top_2_batch_corr_with_prior[idx] >= 1:
                        total_top_2_batch_corr_with_prior[idx] += 1
                if sum(batch_corr) > 0:
                    accurate += 1
#                 print(total_batch_corr)
                    
                    
            img_data = []
        

        if (index+1) % 1000 == 0: 
            print(f"accuracy for {index+1} images top k patches: {[f'{((x/(index+1))*100):2f}%' for idx, x in enumerate(total_batch_corr)]}")
            print(f"accuracy for top 2 from top 1 and next top 19: {[f'{((x/(index+1))*100):2f}%' for idx, x in enumerate(total_top_2_batch_corr)]}")
            print(f"accuracy for top 2 from top 1 and next top 19(prior): {[f'{((x/(index+1))*100):2f}%' for idx, x in enumerate(total_top_2_batch_corr_with_prior)]}")
            print(f"Accuracy if any of the patch was right: {accurate*100/(index+1)}%")
        
    top_k_accuracy = [f'{((x/(index+1))*100):2f}%' for idx, x in enumerate(total_batch_corr)]
    top_2_accuracy = [f'{((x/(index+1))*100):2f}%' for idx, x in enumerate(total_top_2_batch_corr)]
    top_2_accuracy_with_prior = [f'{((x/(index+1))*100):2f}%' for idx, x in enumerate(total_top_2_batch_corr_with_prior)]
    any_patch_correct_accuracy = accurate/len(val_data)
    print(f"OULU test took {time.time() - start}")
    print(f"accuracy for {index+1} images top k patches: {[f'{((x/(index+1))*100):2f}%' for idx, x in enumerate(total_batch_corr)]}")
    print(f"accuracy for top 2 from top 1 and next top 19: {[f'{((x/(index+1))*100):2f}%' for idx, x in enumerate(total_top_2_batch_corr)]}")
    print(f"accuracy for top 2 from top 1 and next top 19(prior): {[f'{((x/(index+1))*100):2f}%' for idx, x in enumerate(total_top_2_batch_corr_with_prior)]}")
    print(f"Accuracy if any of the patch was right: {accurate*100/len(val_data)}%")
    return correct_indices, top_k_accuracy, total_batch_corr, top_2_accuracy, any_patch_correct_accuracy, top_2_accuracy_with_prior 


args    = get_arguments()
im_size = args.im_size
step_size = 1
length = int(((96 - im_size)/step_size) + 1)
gpus = args.gpu
gpus = [int(x.strip()) for x in args.gpu.split(",")]
print(gpus)
model = load_model(args, im_size, gpus)
transform = transforms.Compose([transforms.ToTensor()])
n_samples = 400
REAL,SPOOF = oulu(200)
spoof_dataset = PatchTopDataset(SPOOF, transform=transform, color_mode='rgb', im_size=im_size, phase='test')
live_dataset = PatchTopDataset(REAL, transform=transform, color_mode='rgb', im_size=im_size, phase='test')
spoof_dataloader = torch.utils.data.DataLoader(
    spoof_dataset, batch_size=1, shuffle=False, num_workers=8)

live_dataloader = torch.utils.data.DataLoader(
    live_dataset, batch_size=1, shuffle=False, num_workers=8)

total_combined = 0
correct_combined = 0



correct_indices_spoof, spoof_accuracy, all_indices_spoof, total_combined, correct_combined, spoof_preds = count_correct_patches(spoof_dataloader, total_combined, correct_combined, spoof=True, size=(3, im_size, im_size))
correct_indices_live, live_accuracy,  all_indices_live, total_combined, correct_combined, live_preds = count_correct_patches(live_dataloader, total_combined, correct_combined, spoof=False, size=(3, im_size, im_size))
print(f"Accuracy for live {live_accuracy:.2f}%")
print(f"Accuracy for spoof {spoof_accuracy:.2f}%")

def get_patch_accuracy(correct_indices, all_indices):
    patch_wise_accuracy = {key: (correct_indices[key]/all_indices[key]) *
                           100 if all_indices[key] != 0 else 0 for key in range(0, length*length)}
    return np.array([*patch_wise_accuracy.values()]).reshape(length, length)

spoof_accuracy = get_patch_accuracy(correct_indices_spoof, all_indices_spoof)
live_accuracy = get_patch_accuracy(correct_indices_live, all_indices_live)

spoofs = np.array([*correct_indices_spoof.values()])
lives = np.array([*correct_indices_live.values()])

combined_correct = np.add(spoofs, lives)
all_indices = np.add(np.array([*all_indices_spoof.values()]),
                     np.array([*all_indices_live.values()]))

combined_accuracy = get_patch_accuracy(combined_correct, all_indices)
os.makedirs(f'patch_dicts/{im_size}')
np.save(f'patch_dicts/{im_size}/OULU_spoof.npy', spoof_accuracy)
np.save(f'patch_dicts/{im_size}/OULU_combined.npy', combined_accuracy)
np.save(f'patch_dicts/{im_size}/OULU_live.npy', live_accuracy)

# spoof_accuracy = np.load('patch_dicts/spoof.npy')
# combined_accuracy = np.load('patch_dicts/combined.npy')
# live_accuracy = np.load('patch_dicts/live.npy')
# with open('patch_dicts/spoof.npy', 'w') as file:
#     json.dump(spoof_accuracy, file)

# with open('patch_dicts/live.npy', 'w') as file:
#     json.dump(live_accuracy, file)

# with open('patch_dicts/combined.npy', 'w') as file:
#     json.dump(combined_accuracy, file)


# oulu_data_path = "/home/ec2-user/SageMaker/dataset/OULU/cropped_faces/112x112"
# correct_indices_oulu, oulu_top_k_accuracy, total_batch_corr_oulu, oulu_top_2_accuracy, oulu_any_patch_accuracy, oulu_top_2_accuracy_with_prior = run_on_oulu(oulu_data_path, combined_accuracy, number_of_patches=5)

# oulu = {"correct_indices": correct_indices_oulu,
#         "top_k_acc": oulu_top_k_accuracy,
#         "total_corr": total_batch_corr_oulu,
#         "top_2_acc": oulu_top_2_accuracy,
#         "top_2_acc_prior": oulu_top_2_accuracy_with_prior,
#         "any_patch_acc": oulu_any_patch_accuracy}

# np.save(f'patch_dicts/oulu_{n_samples}_high.npy', np.array(oulu))
# oulu_accuracy = get_patch_accuracy(correct_indices_oulu, all_indices_oulu)
# print(oulu_accuracy.shape)
# np.save('patch_dicts/oulu_accurracy.npy', np.array(oulu_accuracy))
# np.save('patch_dicts/correct_indices_oulu.npy', correct_indices_oulu)
# np.save('patch_dicts/all_indices_oulu.npy', all_indices_oulu)
# np.save('patch_dicts/total_batch_corr_oulu.npy', total_batch_corr_oulu)
# with open('patch_dicts/oulu.npy', 'w') as file:
#     json.dump(oulu_accuracy, file)
# plot_heatmap(oulu_accuracy, "oulu")
plot_heatmap(spoof_accuracy, "spoof")
plot_heatmap(live_accuracy, "live")
plot_heatmap(combined_accuracy, "combined")
