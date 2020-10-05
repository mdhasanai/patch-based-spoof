
import os
import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))
sys.path.append(path.join(path.dirname(__file__), '../..'))


import cv2
import h5py
import glob
import time
import math
import torch
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F
from skimage.util.shape import view_as_windows
from facerec.retinaface_detection import RetinaDetector


from dataloaders.patch_dataloader import PatchDataset
from dataloaders.patch_dataloader import PatchDataset as SpoofDataset







device_ids= [i for i in range(5, 8)]


class New_PatchDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, color_mode=['rgb'], im_size=96, patch_size=None, phase='train', training_type='None'):
        self.fnames = path
        self.color_mode = color_mode
        self.im_size = im_size
        self.transform = transform
        self.patch_size = patch_size
        self.phase = phase
        self.training_type = training_type
        self.detector = RetinaDetector()
        
        self.count = 0
    

    def __getitem__(self, idx):
        
        im, label = self.create_single_sample(idx)
    
        return im, label

    def __len__(self):
        return len(self.fnames)
    
        
    def create_single_sample(self, idx):
        
        im_path = self.fnames[idx]

        label = self.fnames[idx].split("/")#[-2]

        img = cv2.imread(im_path)
        img = cv2.resize(img, (224, 224))

        imgs = {}
        for mode in self.color_mode:
            imgs[mode] = cv2.cvtColor(img, eval(f'cv2.COLOR_BGR2{mode.upper()}'))
        
        imgs = self.getting_patches(imgs, im_path)
        if "spoof" in label or "device" in label or "print" in label or "print2" in label or "video-replay2" in label:
            ground = torch.ones(1)
        elif "print1" in label or "video-replay1" in label:
            ground = torch.ones(1)
        elif "live" in label or "real" in label:
            ground = torch.zeros(1)
#         else:
#             print(label)
        return imgs, ground
    
    def getting_patches(self, img_patch, im_path_name):

        
        new_im = img_patch['rgb']
        
        y1,y2,x1,x2 = self.getting_middle_path(im_path_name)
        new_im = new_im[y1:y2,x1:x2,:]
        new_im_m_tr = self.transform(new_im)
        
        return new_im_m_tr
    

    def divide_single_img_into_patches(self, img, size=(224, 224), patch_size=(48, 48, 3), step=1):

        img = cv2.resize(img, size)
        #             print(img.shape, len(patch_size))
        patch_grid = view_as_windows(img, patch_size, step)
        #         print(patch_grid.shape)
        return patch_grid
    
    def get_another_landmark(self):
        img = np.array([[[ 73.25691 , 106.048355],
                        [149.7507  , 105.70673 ],
                        [111.53898 , 141.779   ],
                        [ 79.02094 , 186.59473 ],
                        [141.23201 , 186.73901 ]]], dtype=np.float32)
        
        return img
    
    def getting_middle_path(self,im_path):
        """
            landmark 0: eye-1     (x,y)
            landmamrk 1: eye-2    (x,y)
            landmark 2: nose      (x,y)
            landmark 3: lip-left  (x,y)
            landmar 4: lip-right  (x,y)
        """
        
        # extracting landmarks
        im, faces, landmarks = self.detector.infer(im_path, resize=[224,224])
        
        if np.array(landmarks).any() == False:
            landmarks = self.get_another_landmark()
            with open("not_found.txt","a") as file:
                file.write(f"{im_path}\n")

        patches = []
        
        half = self.im_size//2 # for 96, half = 48
        
        middle_point_x = (landmarks[0][0][0]+landmarks[0][1][0]) //2
        middle_point_x = (middle_point_x+landmarks[0][2][0]) // 2
        
        middle_point_y = (landmarks[0][0][1]+landmarks[0][1][1])//2 
        middle_point_y = (middle_point_y+ landmarks[0][2][1]) //2
        
        #middle_point_x -=10
       # middle_point_y -=15
        

        if middle_point_y - half < 0:
            y1 = 0
            y2 = self.im_size
        elif middle_point_y + half > im.shape[0]:
            y2 = im.shape[0]
            y1 = im.shape[0] - self.im_size
        else:
            y1 = int(middle_point_y) - half
            y2 = int(middle_point_y) + half
        if middle_point_x - half < 0:
            x1 = 0
            x2 = self.im_size
        elif middle_point_x + half > im.shape[1]:
            x2 = im.shape[0]
            x1 = im.shape[0] - self.im_size
        else:
            x1 = int(middle_point_x) - half
            x2 = int(middle_point_x) + half
        
        return y1,y2, x1,x2
    

class ScoreDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, im_size=96, modeltype="patch", top_patch=[36, 36], colormode=['rgb']):
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
        im = self.create_single_sample(idx)
        
        label = self.fnames[idx].split("/")
        
        if "spoof" in label or "device" in label or "print" in label or "print2" in label or "video-replay2" in label:
            ground = torch.ones(im.size(0))
        elif "live" in label or "real" in label:
            ground = torch.ones(im.size(0))
        return im, ground #self.fnames[idx]
    
    
    def __len__(self):
        return len(self.fnames)
    
    def create_single_sample(self, idx):
#         print(f'{self.fnames[idx]}')
        img = cv2.imread(self.fnames[idx])
        images = {}
        if self.modeltype == "deeppix" :
            if len(self.colormode) > 1:
                for mode in self.colormode:
                    im = cv2.cvtColor(img, eval(f"cv2.COLOR_BGR2{mode.upper()}"))
                    im = cv2.resize(im, (224, 224))
                    im = self.transform(im)
                    images[mode] = im
                return images
            else:
                im = cv2.cvtColor(img, eval(f"cv2.COLOR_BGR2{self.colormode[0].upper()}"))
                img = cv2.resize(img, (224, 224))
                return self.transform(im)    
            
        img = cv2.resize(img, (224, 224))

        imgs = cv2.cvtColor(img, eval(f'cv2.COLOR_BGR2RGB'))
        imgs = self.get_batch_imgs(imgs)
        return imgs
    
    
    
    def make_patches(self, img_patch, length, size=96):

        random_indices = np.random.choice(np.arange(length), 20)

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

        size = 224 if self.im_size == 96 else 96
        
        imgs = self.get_random_patches(img_patch, size=size)
        
        return imgs
    
    def get_random_patches(self, img_patch, size):
        
        length = int(((size - self.im_size)/1) + 1)
        
        
        patches = self.make_patches(img_patch, length, size)
           
        imgs = torch.zeros((20, 3, self.im_size, self.im_size))

        for idx in range(20):
            im_tr = self.transform(patches[idx])
            imgs[idx] = im_tr
        return imgs
    
    def divide_single_img_into_patches(self, img, size=(224, 224), patch_size=(96, 96, 3), step=1):

        img = cv2.resize(img, size)

        patch_grid = view_as_windows(img, patch_size, step)

        return patch_grid
    

def eval_model( model, dataloader, criterion, best_acc=-1):

    total = 0
    running_corrects = 0
    running_loss = 0.0
    tq = tqdm(dataloader, desc="Validating")

    torch.cuda.empty_cache()

    for idx, (im, label) in enumerate(tq):
        im = im.reshape(-1, im.size(2), im.size(3), im.size(3))
        label = label.reshape(-1)
        im = im.cuda(device_ids[0])
        label = label.type(torch.LongTensor).cuda(device_ids[0])
        out = model(im)
        outputs = model(im)
        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, label)

        running_loss += loss.item() * im.size(0)
        running_corrects += torch.sum(preds == label.data)
        total += im.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total
    if epoch_acc > best_acc:
        best_acc = epoch_acc

    print(total, running_corrects, running_loss)
    print(f'Validation Loss: {epoch_loss:.4f} -- Acc: {epoch_acc * 100:.4f}%')
    return best_acc

def get_patch_size(img_size):
    if img_size==96:
        return 20
    elif img_size==48:
        return 30
    elif img_size==24:
        return 40
    elif img_size==96:
        return 50
       
def laod_keras_model(path=None):
    
    import tensorflow as tf
    from tensorflow import keras
    
    if path is None:
        keras_model = tf.keras.models.load_model('../ckpts/patch_based_cnn/96/our_data_best.h5')
    else:
        keras_model = tf.keras.models.load_model(path)
    
    return keras_model

def infer_keras_model(keras_model, imgs, labels):
    # batch []
    imgs = imgs.cpu().detach().numpy()
    imgs = imgs.reshape(imgs.shape[0],imgs.shape[2],imgs.shape[2],-1)
    labels = labels.cpu().detach().numpy()
    labels = labels.reshape(1,-1)
    
    output = keras_model.predict_on_batch(imgs)
    preds = np.array(np.argmax(output, axis=1)).reshape(1, -1)
    
    correct = np.sum(preds==labels)
    return  preds, correct



def eval_data(model, dataloader, batch_size, is_keras_model=False, keras_model_name=None):
    
    if is_keras_model:
        if keras_model_name == None:
            keras_model = laod_keras_model()
        else:
            keras_model = laod_keras_model(keras_model_name)
    
    middle_accuracy = 0
    
    #model = model.double()
    model = model.eval()
    
    total_steps = 0
    
    tq = tqdm(dataloader)

    for idx, (imgs, label) in enumerate(tq):

        if is_keras_model:
            preds, correct    = infer_keras_model(keras_model, imgs, label)
            running_corrects = correct 
        else:
            img = imgs.cuda(0)
            outputs = model(img)
            probs, preds = torch.max(outputs, 1)
            label = label.reshape(-1)
            
            #running_corrects = torch.sum(preds.cpu() == label.data)
            running_corrects = np.sum(preds.cpu().detach().numpy() == label.data.detach().numpy())
        
        middle_accuracy += running_corrects #.item()
        
            
        total_steps += img.size(0)
        
        temp_acc = middle_accuracy/total_steps
        
        tq.set_postfix(iter=idx, acc=temp_acc)
    
    total_accuracy = middle_accuracy/total_steps#(batch_size*total_steps)
    print(f"Total num of acc: {middle_accuracy}| Total iter {total_steps}")
    print(f"Total eval acc: {total_accuracy}")
    
    return total_accuracy


def load_model(im_size, model_name="oulu", modeltype="patch"):
    if modeltype == "patch":
        from models.patch_based_cnn.model import PatchModel
        model = PatchModel(im_size)
        model = torch.nn.DataParallel(model, device_ids)
        model_path = f"../ckpts/patch_based_cnn/{im_size}/{model_name}_model.pth"
        model.load_state_dict(torch.load(model_path))
        
    elif modeltype == "deeppix":
        model_path = f'../ckpts/deeppix/six_channel/proto4/proto4-best.pt'
        model = torch.load(model_path)
        model = torch.nn.DataParallel(model.module, device_ids)
        
    model = model.cuda(device_ids[0])
    model.eval()
    
    return model
def nagod():
    test_path = "/home/ec2-user/SageMaker/dataset/nogod_frames/"
    test_images = glob.glob(f"{test_path}/***/**/*")
    print("nagod Replay test set : ",len(test_images))
    return test_images
    
    
    
def combined_dataset(hard_protocol="Protocol_4"):
    
    
    mobile_train_path = "/home/ec2-user/SageMaker/dataset/spoof-data/mobile_replay"
    msu_train_path = "/home/ec2-user/SageMaker/dataset/spoof-data/msu"
    oulu_train_path = "/home/ec2-user/SageMaker/dataset/spoof-data/oulu"
    
    oulu_train = glob.glob(f"{oulu_train_path}/{hard_protocol}/Train/**/*")
    oulu_val   = glob.glob(f"{oulu_train_path}/{hard_protocol}/Dev/**/*")
    
    #print(len(oulu_train),len(oulu_val))
    
    msu_train = glob.glob(f"{msu_train_path}/Train/**/*")
    msu_val   = glob.glob(f"{msu_train_path}/Test/**/*")
    
    #print(len(msu_train),len(msu_val))
    
    mobile_replay_train = glob.glob(f"{mobile_train_path}/train/***/**/*")
    mobile_replay_val   = glob.glob(f"{mobile_train_path}/test/***/**/*")
    
    #print(len(mobile_replay_train),len(mobile_replay_val))
    
    train_imgs = oulu_train + msu_train + mobile_replay_train
    val_imgs   = oulu_val +  msu_val + mobile_replay_val
    
    print(f"Total combined images | Training: {len(train_imgs)} | Valid: {len(val_imgs)}")
    return val_imgs#{"train": train_imgs, "val": val_imgs}

def mobile_replay(proto=None):
    
    #train_path = "../../../dataset/spoof-data/mobile_replay/train"
    test_path = "../../dataset/spoof-data/mobile_replay/test"
    #train_imgs = glob.glob(f"{train_path}/***/**/*")
    test_images = glob.glob(f"{test_path}/***/**/*")
    print("Mobile Replay test set : ",len(test_images))
    return test_images#{"train": train_imgs, "test": val_imgs}

def oulu(hard_protocol='Protocol_4'):
    
    oulu_train_path = "/home/ec2-user/SageMaker/dataset/spoof-data/oulu"
    
    test_imgs = glob.glob(f"{oulu_train_path}/{hard_protocol}/Test/**/*") 

    print("OULU data test vs val: ",len(test_imgs))
    return test_imgs

def msu(proto=None):
    
    train_path = "../../dataset/spoof-data/msu/Train"
    test_path = "../../dataset/spoof-data/msu/Test"
    train_imgs = glob.glob(f"{train_path}/**/*")
    test_imgs = glob.glob(f"{test_path}/**/*")
    print("MSU data test: ",len(test_imgs))
    return test_imgs #{"train": train_imgs, "test": val_imgs}


parser = argparse.ArgumentParser(description='PASS NECESSARY ARGS')
parser.add_argument('--eval_dataset', help='select evaluation dataset', default="oulu")
parser.add_argument('--model_type', help='test on msu or oulu or mobile reply', default="oulu")
parser.add_argument('--image_size', help='12, 24, 48, 96', default=48)
parser.add_argument('--batch_size', help='batch size', default=128)
parser.add_argument('--protocol', help='batch size', default="Protocol_1")
parser.add_argument('--data_protocol', help='batch size', default="Protocol_1")
parser.add_argument('--get_acc', help='1 for kypoint and 2 for origina model', default=1)

args = vars(parser.parse_args())
transform = transforms.Compose([transforms.ToTensor()])
def run():
    eval_dataset = args["eval_dataset"]
    model_type = args["model_type"]
    image_size = int(args["image_size"])
    batch_size = int(args["batch_size"])
    get_acc = int(args["get_acc"])
    
    if model_type == "oulu":
        model_type = model_type + "_" + str(args["protocol"]) #args.protocol
    
    model = load_model(image_size,model_type)
    
    with open("./middle_patch/keypoint_results_new.txt","a") as file:

        if get_acc == 1:
            print(f"Evaluation Data: {eval_dataset}")
            if eval_dataset == "msu":
                images = msu()
            elif eval_dataset == "oulu":
                protocol = str(args["data_protocol"]) 
                eval_dataset = eval_dataset+"-"+protocol
                images = oulu(protocol)
            elif eval_dataset == "mobile_replay":
                images = mobile_replay()
            elif eval_dataset == "combined":
                images = combined_dataset()
            elif eval_dataset=="nagod":
                images = nagod()
                
            print("TYpey: 1")
            dataset = New_PatchDataset(images, transform=transform, im_size=image_size, color_mode=['rgb'])
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            print(f"Length Data: {len(dataloader)}")
            spoof_accuracy = eval_data(model, dataloader, batch_size)
            print(f"Accuracy: {spoof_accuracy*100}")
            
            file.write(f"Trained On: {model_type} | Test dataset: {eval_dataset} | Patch size: {image_size} | Accuracy: {spoof_accuracy*100} \n \n")

        elif get_acc==2:
            if eval_dataset == "msu":
                images = msu()
            elif eval_dataset == "oulu":
                protocol = str(args["data_protocol"]) 
                eval_dataset = eval_dataset+"-"+protocol
                images = oulu(protocol)
            elif eval_dataset == "mobile_replay":
                images = mobile_replay()

            loss = torch.nn.NLLLoss()

            patch_size=get_patch_size(image_size)

            dataset = SpoofDataset(images, transform=transform, im_size=image_size, patch_size=patch_size , color_mode=['rgb'])
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=64)

            results = eval_model(model, dataloader, loss)

run()
        

    
