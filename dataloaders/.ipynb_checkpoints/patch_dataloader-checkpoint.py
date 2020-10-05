import time
import math
import torch
import numpy as np
import cv2
import random
import glob
import h5py
from skimage.util.shape import view_as_windows
from torchvision import transforms
import sys


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, csv_data, transform=None, color_mode=['rgb'], im_size=48, no_patch=None, phase='train',top_patch=[66,60]):
        #self.csv_data      = csv_data
        self.fnames        = csv_data['file_name'].tolist()
        self.labels        = csv_data['label'].tolist()
        self.color_mode    = color_mode
        self.im_size       = im_size
        self.transform     = transform
        self.no_patch      = no_patch
        self.phase         = phase
        self.top_patch     = top_patch
        
    def __getitem__(self, idx):
        im, label, acc_label  = self.create_single_sample(idx)
        return im, label, acc_label
        
    def blacken_patch(self, img, top_patches):
        black = np.zeros((self.im_size, self.im_size, 3))
        for top_patch in top_patches:
            prob = random.uniform(0, 1)
            if prob > .5:
                img[top_patch[0]:top_patch[0] + self.im_size, top_patch[1]: top_patch[1] + self.im_size, :] = black
        return img
    
    def create_single_sample(self, idx):
        im_path = self.fnames[idx]
        label   = self.fnames[idx].split("/")[-3]
        img     = cv2.imread(im_path)
        img     = cv2.resize(img, (224, 224))

        imgs = {}
        for mode in self.color_mode:
            imgs[mode] = cv2.cvtColor(img, eval(f'cv2.COLOR_BGR2{mode.upper()}'))
        imgs = self.get_batch_imgs(imgs)

        if  label=="spoof":
            ground = torch.ones(imgs.size(0))
            acc_label = torch.ones(1)
        elif label=="real":
            ground = torch.zeros(imgs.size(0))
            acc_label = torch.zeros(1)
        return imgs, ground, acc_label
    

    def divide_single_img_into_patches(self, img, size=(112, 112), no_patch=(96, 96, 3), step=1):
        img = cv2.resize(img, size)
        patch_grid = view_as_windows(img, no_patch, step)
        return patch_grid

    def get_calculated_patches(self, img_patch):
        iterations = 6 - int(math.log((self.im_size // 3), 2))
        total_imgs = sum([4 ** i for i in range(iterations)])
        imgs = torch.zeros((total_imgs, len(self.color_mode) * 3, self.im_size, self.im_size))
        step = 24
        rows = 0
        cols = 0
        for i in range(iterations):
            start = rows
            end = cols
            while end + self.im_size < 96:
                while start + self.im_size < 96:
                    for mode in self.color_mode:
                        im = img_patch[mode][start:start + self.im_size, end:end + self.im_size, :]
 
                        tensor = self.transform(im)
                        imgs[i] = tensor
                    start += self.im_size
            end += self.im_size
            start = 0
        return imgs
    
    
    def get_top_n(self, accuracy, n=2, threshold=95):
        top_n_indices = []
        prev = 100
        i = 1
        data = np.unique(-np.sort(-accuracy.flatten()))
        while True:
            top_n_val = data[-i]
            if top_n_val == prev:
                conitnue
            lis = np.argwhere(np.logical_and(accuracy >= top_n_val, accuracy<prev))
            top_n_indices.extend(lis)
            if top_n_val <= threshold:
                break
            prev = top_n_val
            i += 1
        return top_n_indices
    
    def make_patches(self, img_patch, length, size=96):
        patch_96x96 = {}
        random_indices = np.random.choice(np.arange(length), self.no_patch)
        concats = []
        for mode in self.color_mode:
            patch_96x96[mode] = self.divide_single_img_into_patches(img_patch[mode], size=(size, size),
                                                          no_patch=(self.im_size, self.im_size, 3)).reshape(
            (-1, self.im_size, self.im_size, 3))
            concats.append(torch.from_numpy(patch_96x96[mode][random_indices]))
        return concats
    
    
    def get_random_patches(self, img_patch, size):
        length = int(((size - self.im_size)/1) + 1)
        concats = self.make_patches(img_patch, length, size)
           
        im = torch.cat(concats, 3)
        imgs = torch.zeros((self.no_patch, im.size(3), self.im_size, self.im_size))
        for idx in range(self.no_patch):
            im_tr = self.transform(im[idx].numpy())
            imgs[idx] = im_tr
        return imgs


    def get_batch_imgs(self, im):
        img_patch = {}
        if self.im_size != 96:
            for mode in self.color_mode:
                img_patch[mode] = im[mode][self.top_patch[0]:self.top_patch[0]+96, self.top_patch[1]:self.top_patch[1]+96, :]
        else:
            img_patch = im
        size = 224 if self.im_size == 96 else 96
        if self.no_patch is not None:
            imgs = self.get_random_patches(img_patch, size=size)
        else:
            imgs = self.get_calculated_patches(img_patch)
        return imgs
    
    def __len__(self):
        return len(self.fnames)