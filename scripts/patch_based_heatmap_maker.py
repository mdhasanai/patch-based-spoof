import glob
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

from tqdm import tqdm #images done so far

# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1' 

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.30)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# K.set_session(sess)
class PadSameConv(torch.nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride, pad_size):
        super(PadSameConv, self).__init__()
        self.pad_size = pad_size
        self.conv = torch.nn.Conv2d(in_size, out_size, kernel_size, stride)

    def forward(self, x):
        out = F.pad(x, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), 'circular')
        out = self.conv(out)
        return out


class ChannelBasedLayer(torch.nn.Module):
    def __init__(self, input_size, depth_size):
        super(ChannelBasedLayer, self).__init__()
        output_size = 150
        step = 50
        modules = []
        cnt = depth_size // 50 - 2
        for _ in range(cnt):
            modules.append(PadSameConv(input_size, output_size, kernel_size=3, stride=1, pad_size=1))
            modules.append(torch.nn.BatchNorm2d(output_size))
            modules.append(torch.nn.MaxPool2d(2, 2))
            modules.append(torch.nn.ReLU())
            input_size += step
            output_size += step

        self.sequential = torch.nn.Sequential(*modules)

    #         print(self.sequential)

    def forward(self, x):
        return self.sequential(x)


class PatchModel(torch.nn.Module):
    def __init__(self, im_size):
        self.depth_size = self.get_depth_size(im_size)
        super(PatchModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 50, kernel_size=5, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(50)
        self.maxpool1 = torch.nn.MaxPool2d(2, 2)

        self.conv2 = torch.nn.Conv2d(50, 100, kernel_size=3, stride=1)
        self.bn2 = torch.nn.BatchNorm2d(100)
        self.maxpool2 = torch.nn.MaxPool2d(2, 2)

        self.conv3 = ChannelBasedLayer(100, self.depth_size)

        self.fc1 = torch.nn.Linear(self.depth_size * 3 * 3, 1000)
        self.bn6 = torch.nn.BatchNorm1d(1000)
        self.fc2 = torch.nn.Linear(1000, 400)
        self.bn7 = torch.nn.BatchNorm1d(400)
        self.fc3 = torch.nn.Linear(400, 2)

        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.relu = torch.nn.ReLU()

    def get_depth_size(self, im_size):
        depth_size = int(math.log(im_size // 3, 2)) * 50
        return depth_size

    def forward(self, x):
        out = F.pad(x, (2, 2, 2, 2), 'circular')
        out = self.bn1(self.conv1(out))
        out = self.maxpool1(out)
        out = self.relu(out)
        ### 50 * 24 * 24 | 12 * 12

        out = F.pad(out, (1, 1, 1, 1), 'circular')
        out = self.maxpool2(self.bn2(self.conv2(out)))
        out = self.relu(out)
        ### 100 * 12 * 12 | 6 * 6

        out = self.conv3(out)

        ### 150 * 6 * 6 | 3 * 3

        out = out.view(-1, self.depth_size * 3 * 3)

        out = self.bn6(self.fc1(out))
        out = self.relu(out)

        out = self.bn7(self.fc2(out))
        out = self.relu(out)

        out = self.fc3(out)
        out = self.logsoft(out)

        return out
im_size = 48
step_size = 1
length = int(((96 - im_size)/step_size) + 1)
#model_path = f'../ckpts/patch_based_cnn/{im_size}/model_rgb.pth'
model_path = f'../ckpts/patch_based_cnn/{im_size}/mobile_replay_model.pth'
#data_path = "../../dataset/msu/dataset/Test/"
#data_path = "../../dataset/spoof-data/msu/Test/"
data_path = "../../dataset/spoof-data/mobile_replay/test"
model = PatchModel(im_size)
model = torch.nn.DataParallel(model, device_ids=[3, 4, 5, 6]) #[0, 1, 2, 3]
model.load_state_dict(torch.load(model_path))
model.cuda(3)
spoof_path = data_path + "/spoof/"
live_path = data_path + "/real/"
transform = transforms.Compose([transforms.ToTensor()])
n_samples = 50000


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

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, color_mode='rgb', im_size=48, patch_size=None, create_data=True, phase='train', datatype='msu'):
        self.fnames = glob.glob(f"{path}/**/*.*", recursive=True)
        self.fnames = list(np.random.choice(self.fnames, n_samples))
        self.color_mode = color_mode
        self.im_size = im_size
        self.transform = transform
        self.patch_size = patch_size
        self.phase = phase
        self.datatype = datatype
#         if create_data:
#             self.im_data, self.labels = self.create_data()
#             self.save_data()
#         else:
#             self.im_data, self.labels = self.load_data()

    def __getitem__(self, idx):
#         im, label = self.im_data[idx], self.labels[idx]
#         print(idx)
        im, label = self.create_single_sample(idx)
#         print(label)
        return im, label

    def __len__(self):
        return len(self.fnames)
    
    def load_data(self):
        hf = h5py.File(f'saved_data/10000_{self.phase}_{self.im_size}_sized_im_data.h5', 'r')
        im_data = hf.get('im_data')
        labels = hf.get('labels')
        hf.close()
        return im_data, labels
    
    def save_data(self):
        hf = h5py.File(f'saved_data/{self.phase}_{self.im_size}_sized_im_data.h5', 'w')
        hf.create_dataset('im_data', data=self.im_data)
        hf.create_dataset('labels', data=self.labels)
        hf.close()
    
    def create_single_sample(self, idx):
        
        im_path = self.fnames[idx]

        label = self.fnames[idx].split("/")[-3]

        img = cv2.imread(im_path)

        img = cv2.cvtColor(img, eval(f'cv2.COLOR_BGR2{self.color_mode.upper()}'))
#         patch_grid = self.divide_single_img_into_patches(img, patch_size=(96, 96, 3), step=1).reshape((-1, 96, 96, 3))
            #         print(patch_grid.shape)
        
        imgs = self.get_batch_imgs(img, self.datatype)
        if label == "spoof" in label or "device" in label or "print" in label or "video-replay" in label:
            label = torch.ones(imgs.size(0))
#             label = torch.tensor(1)
        elif label == "live" or label == "real":
            label = torch.zeros(imgs.size(0))
#             label = torch.tensor(0)
#         print(label)
        return imgs, label

    
    def create_data(self):
        start = time.time()
        im_data = torch.zeros((1000, length*length, 3, self.im_size, self.im_size))
        labels = torch.zeros(1000, length*length)
        
        for idx in range(len(self.fnames)):
#             print(self.fnames[idx])
            imgs, label = self.create_single_sample(idx)
            im_data[idx] = imgs
            labels[idx] = label
            if idx % 100 == 0:
#                 print(f"{idx+1} images are loaded so far!!")
                print(f"It took {time.time() - start} to load!")
            if (idx+1) % 1000 == 0:
#                 print(idx+1)
                break
        return im_data, labels

    def divide_single_img_into_patches(self, img, size=(224, 224), patch_size=(96, 96, 3), step=1):

        img = cv2.resize(img, size)
        #             print(img.shape, len(patch_size))
        patch_grid = view_as_windows(img, patch_size, step)
        #         print(patch_grid.shape)
        return patch_grid

    def get_calculated_patches(self, img_patch):
        iterations = 6 - int(math.log((self.im_size // 3), 2))
        total_imgs = sum([4 ** i for i in range(iterations)])
        imgs = torch.zeros((total_imgs, 3, self.im_size, self.im_size))
        step = 24
        rows = 0
        cols = 0
        for i in range(iterations):
            start = rows
            end = cols
            while end + self.im_size < 96:
                while start + self.im_size < 96:
                    im = img_patch[start:start + self.im_size, end:end + self.im_size, :]
                    tensor = self.transform(im)
                    imgs[i] = tensor
                    start += self.im_size
            end += self.im_size
            start = 0

    def get_random_patches(self, img_patch):
        patch_96x96 = self.divide_single_img_into_patches(img_patch, size=(96, 96),
                                                          patch_size=(self.im_size, self.im_size, 3)).reshape(
            (-1, self.im_size, self.im_size, 3))
#         random_20 = np.random.choice(np.arange(patch_96x96.shape[0]), 20)
#         selected_imgs = patch_96x96[random_20]

#         imgs = torch.zeros((20, 3, self.im_size, self.im_size))
#         for idx in range(self.patch_size):
#             imgs[idx] = self.transform(selected_imgs[idx])
         
        imgs = torch.zeros((patch_96x96.shape[0], 3, self.im_size, self.im_size))

        for i in range(patch_96x96.shape[0]):
            imgs[i] = self.transform(patch_96x96[i])
# #             print(i)
        return imgs
    def get_oulu_patches(self, img_patch):
        accuracy = np.load(f"patch_dicts/{self.im_size}/combined.npy")
        top_indices = get_top_n(accuracy, 1)
        imgs = torch.zeros((len(top_indices), 3, self.im_size, self.im_size))
        for i in range(len(top_indices)):
            imgs[i] = self.transform(img_patch[top_indices[i][0]:top_indices[i][0]+self.im_size, top_indices[i][1]:top_indices[i][1]+self.im_size,:])
            
        return imgs

    def get_batch_imgs(self, im, datatype):
        im = cv2.resize(im, (224, 224))
        img_patch = im[66:66+96, 60:60+96, :]
        if self.datatype == 'msu':
            imgs = self.get_random_patches(img_patch)
        else:
            imgs = self.get_oulu_patches(img_patch)

        return imgs



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
        imgs = imgs.cuda(3)
        label = label.type(torch.LongTensor).cuda(3)
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
        imgs = imgs.cuda(3)
        label = label.type(torch.LongTensor).cuda(3)

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


spoof_dataset = PatchDataset(spoof_path, transform=transform, color_mode='rgb', im_size=im_size, patch_size=None, create_data=True, phase='validation')
live_dataset = PatchDataset(live_path, transform=transform, color_mode='rgb', im_size=im_size, patch_size=None, create_data=True, phase='validation')
spoof_dataloader = torch.utils.data.DataLoader(
    spoof_dataset, batch_size=1, shuffle=False, num_workers=64)

live_dataloader = torch.utils.data.DataLoader(
    live_dataset, batch_size=1, shuffle=False, num_workers=64)

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
np.save(f'patch_dicts/{im_size}/mobile_spoof.npy', spoof_accuracy)
np.save(f'patch_dicts/{im_size}/mobile_combined.npy', combined_accuracy)
np.save(f'patch_dicts/{im_size}/mobile_live.npy', live_accuracy)

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
