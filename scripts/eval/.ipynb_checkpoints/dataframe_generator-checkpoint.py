import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))
sys.path.append(path.join(path.dirname(__file__), '../..'))
from torchvision import transforms
from torch.nn import functional as F
import math
import torch
import pandas as pd
import math
import glob, cv2
import random
import glob
import h5py
from skimage.util.shape import view_as_windows
from torchvision import transforms
import os
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='PASS NECESSARY ARGS')
parser.add_argument('--train_dataset', help='path to dev.csv is required', default="oulu")
# parser.add_argument('--test', help='path to test.csv is required', required=True)
args = vars(parser.parse_args())
batch_size = 1
im_sizes = [48]#[48, 24]
transform = transforms.Compose([transforms.ToTensor()])
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
path = "../../../dataset/spoof-data/cropped_faces/112x112/Protocol_4/Dev/data"
# path = "../../../dataset/OULU/cropped_faces/112x112/Protocol_4/Test"
top_patch = {48: [36, 36], 24: [49, 55], 12: None, 96: None} #{96: [66, 60]}
device_ids= [i for i in range(4, 8)]
# model_path = f'../../ckpts/deeppix/six_channel/proto1-best.pt'
print(device_ids)
modeltype = "patch"  # patch
train_dataset = args["train_dataset"]
#eval_datasets = ["msu", "oulu", "mobile_replay"]
eval_datasets = ["oulu"]

# eval_datasets = []
class ScoreDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, im_size=48, modeltype="patch", top_patch=[36, 36], colormode=['rgb']):
#         self.fnames = glob.glob(f"{path}/**/*")
        self.fnames = path
        print(len(self.fnames))
#         self.fnames = list(np.random.choice(self.fnames, 5))
        self.im_size = im_size
        self.transform = transform
        self.modeltype = modeltype
        self.top_patch = top_patch
        self.colormode = colormode
        
    def __getitem__(self, idx):
        self.idxx = idx
        im = self.create_single_sample(idx)
#         print(im.shape)
        return im, self.fnames[idx]
    
    
    def __len__(self):
        return len(self.fnames)
    
    def get_paper_size(self):
        label = self.fnames[self.idxx].split("/")
        #print("labels",label)
        if "spoof" in label or "device" in label or "print" in label or "print2" in label or "video-replay2" in label:
            return 8
        elif "print1" in label or "video-replay1" in label:
            return 8
        elif "live" in label or "real" in label:
            return 8
        
    
    def create_single_sample(self, idx):
#         print(f'{self.fnames[idx]}')
        img = cv2.imread(self.fnames[idx])
        images = {}
        if self.modeltype == "deeppix" :
            if len(self.colormode) > 1:
                for mode in self.colormode:
                    im = cv2.cvtColor(img, eval(f"cv2.COLOR_BGR2{mode.upper()}"))
                    im = cv2.resize(im, (112, 112))
                    im = self.transform(im)
                    images[mode] = im
                return images
            else:
                im = cv2.cvtColor(img, eval(f"cv2.COLOR_BGR2{self.colormode[0].upper()}"))
                img = cv2.resize(img, (112, 112))
                return self.transform(im)    
            
            
        img = cv2.resize(img, (112, 112))
        

        imgs = cv2.cvtColor(img, eval(f'cv2.COLOR_BGR2RGB'))
        
        imgs = self.get_batch_imgs(imgs)
        return imgs
    
    
    
    def make_patches(self, img_patch, length, size=48):

        
        random_indices = np.random.choice(np.arange(length), 20)
#         print(random_indices)
        patch_96x96 = self.divide_single_img_into_patches(img_patch, size=(size, size),
                                                          patch_size=(self.im_size, self.im_size, 3)).reshape(
            (-1, self.im_size, self.im_size, 3))

        return patch_96x96[random_indices]
    
    def get_batch_imgs(self, im):
        img_patch = {}
        if self.im_size != 96:
            img_patch = im[66:66+96, 60:60+96, :]
        else:
            img_patch = im
#         if self.im_size == 96:
#             concats = self.make_patches(img_patch, 20, size=224)
            
#             im = torch.cat(concats, 3)
    
#             imgs = torch.zeros((self.patch_size, im.size(3), self.im_size, self.im_size))
#     #         print(imgs.shape)
#             for idx in range(self.patch_size):
#                 im_tr = self.transform(im[idx].numpy())
#                 imgs[idx] = im_tr
#             return imgs
        
    
#         print(self.color_mode)

        size = 224 if self.im_size == 96 else 112
        
        imgs = self.get_random_patches(img_patch, size=size)
        
        return imgs
    
    def get_random_patches(self, img_patch, size):
        
#         selected_imgs = {}
        
        length = int(((size - self.im_size)/1) + 1)
        
        
        patches = self.make_patches(img_patch, length, size)
           

#         print(im.shape)
        get_size = self.get_paper_size()
        #print("get size",get_size)
        imgs = torch.zeros((get_size, 3, self.im_size, self.im_size))
#         print(imgs.shape)
        for idx in range(0,get_size):
            im_tr = self.transform(patches[idx])
            imgs[idx] = im_tr
        return imgs
    def divide_single_img_into_patches(self, img, size=(112, 112), patch_size=(48, 48, 3), step=1):

        img = cv2.resize(img, size)
        #             print(img.shape, len(patch_size))
        patch_grid = view_as_windows(img, patch_size, step)
        #         print(patch_grid.shape)
        return patch_grid

def mobile_replay(proto=None):
    
    train_path = "../../../dataset/spoof-data/mobile_replay/train"
    val_path = "../../../dataset/spoof-data/mobile_replay/test"
    train_imgs = glob.glob(f"{train_path}/***/**/*")
    val_imgs = glob.glob(f"{val_path}/***/**/*")
    return {"train": train_imgs, "test": val_imgs}


def oulu(hard_protocol='Protocol_4'):
    
    oulu_train_path = "/home/ec2-user/SageMaker/dataset/spoof-data/oulu"
    
    train_imgs = glob.glob(f"{oulu_train_path}/{hard_protocol}/Train/**/*") 
    val_imgs   = glob.glob(f"{oulu_train_path}/{hard_protocol}/Dev/**/*") 
    test_imgs  = glob.glob(f"{oulu_train_path}/{hard_protocol}/Test/**/*") 


    print(len(train_imgs), len(test_imgs), len(val_imgs))
    return {"train": train_imgs, "test": test_imgs, "val": val_imgs}

def msu(proto=None):
    
    train_path = "../../../dataset/spoof-data/msu/Train"
    val_path = "../../../dataset/spoof-data/msu/Test"
    train_imgs = glob.glob(f"{train_path}/**/*")
    val_imgs = glob.glob(f"{val_path}/**/*")
    return {"train": train_imgs, "test": val_imgs}



def load_model(im_size, model_name="oulu_Protocol_1", modeltype="patch"):
    if modeltype == "patch":
        from models.patch_based_cnn.model import PatchModel
        model = PatchModel(im_size)
        model = torch.nn.DataParallel(model, device_ids)
        model_path = f"../../ckpts/patch_based_cnn/{im_size}/{model_name}_model.pth"
        model.load_state_dict(torch.load(model_path))
        
    elif modeltype == "deeppix":
        model_path = f'../ckpts/deeppix/six_channel/proto4/proto4-best.pt'
        model = torch.load(model_path)
        model = torch.nn.DataParallel(model.module, device_ids)
        
    model = model.cuda(device_ids[0])
    model.eval()
    
    return model


def infer(imgs, im_path, model):
    if isinstance(imgs, dict):
        for k, v in imgs.items():
            v = v.cuda(device_ids[0])
    else:
        imgs = imgs.cuda(device_ids[0]).squeeze(0)
    #print(imgs.shape)
#     print(model)
    
    outputs = model(imgs)
    #print(outputs)
    return outputs

def get_score(res, modeltype):
    if modeltype == "patch":
#         print(res.shape)
        outs = torch.exp(res.mean(0))
#         print(outs.shape)
        probs = outs[1].data.cpu()
    elif modeltype == "deeppix":
        l_map = res[0].view(res[0].size(0), -1)
        probs = l_map.mean(1)
    return probs

def load_dataset(im_size, dset, phase):
    get_data = globals().get(dset)
    im_data = get_data()
    dataset = ScoreDataset(im_data[phase], transform=transform, im_size=im_size, modeltype=modeltype, top_patch=top_patch[im_size], colormode=['rgb'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=64)
    return dataloader

def get_dataframe(im_size, dset, phase):
    score_df = pd.DataFrame(columns=['frame', 'score'])
    dataloader = load_dataset(im_size, dset, phase)
    model = load_model(im_size)
    
    for idx, (img, im_path) in enumerate(dataloader):
        
        outs = infer(img, im_path, model)
        probs = get_score(outs, modeltype)
        #print(probs)
        #break
        score_df.loc[idx] = [im_path[0], probs.item()]
        if idx % 10 == 0:
            print(f"{(idx + 1)*batch_size} images are done!!! and Prob {probs.item()}")
            
    #score_df.to_csv(f'score_data/patch/{train_dataset}/{phase}/{dset}_{im_size}.csv', index=False, header=False)
    return score_df
    
import os
def main():
    color_mode = 'hsv_ycrcb'
    if modeltype=='patch':
        
        for im_size in im_sizes:
            for dset in eval_datasets:
                for phase in ["test", 'val']:
                    if dset != "oulu" and phase == "val":
                        continue
                    print(f"Inferring for {dset} and {phase}")
                    score_df = get_dataframe(im_size, dset, phase)
                    print(f"Done with {im_size} patches")
                    if dset != "oulu":
                        os.makedirs(f'./score_data/patch/{train_dataset}/val/', exist_ok= True)
                        score_df.to_csv(f'score_data/patch/{train_dataset}/val/{dset}_{im_size}.csv', index=False, header=False)
                    os.makedirs(f'score_data/patch/{train_dataset}/{phase}/', exist_ok= True)
                    score_df.to_csv(f'score_data/patch/{train_dataset}/{phase}/{dset}_{im_size}.csv', index=False, header=False)

    else:
        if not os.path.exists(f'./score_data/{color_mode}'):
            os.makedirs(f'./score_data/{color_mode}')
        score_df = get_dataframe(24)
        score_df.to_csv(f'./score_data/{color_mode}/score-{modeltype}-oulu-proto4-dev-{color_mode}.csv', index=False, header=False)
main()