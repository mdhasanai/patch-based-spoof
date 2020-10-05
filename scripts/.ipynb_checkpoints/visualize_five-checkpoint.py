import os, cv2, sys
sys.path.append("..")
import torch, glob
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.nn import functional as F
from models.patch_based_cnn.model import PatchModel
from random import sample

top_patches = {48: [36, 36], 24: [49, 55]}
wrong = {}
correct = {}
datatypes = ["oulu", "msu"]
path = {"msu": '../../dataset/msu/dataset/Test', "oulu": '../../dataset/OULU/cropped_faces/112x112/Protocol_4/Test'}
batch_size = 192
im_sizes = [48, 24]
transform = transforms.Compose([transforms.ToTensor()])
device_ids= [i for i in range(0, 8)]
modeltype = "patch"
class ScoreDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None, im_size=48, modeltype="patch", top_patch=[36, 36], colormode='rgb'):
        print(data_path)
        self.fnames = glob.glob(f"{data_path}/**/*")
        print(len(self.fnames))
#         self.fnames = list(np.random.choice(self.fnames, 20))
        self.im_size = im_size
        self.transform = transform
        self.modeltype = modeltype
        self.top_patch = top_patch
        self.colormode = colormode
        
    def __getitem__(self, idx):
        im, label = self.create_single_sample(idx)
#         print(im.shape, label)
        return im, self.fnames[idx], label
    
    def __len__(self):
        return len(self.fnames)
    
    def create_single_sample(self, idx):
        im = cv2.imread(self.fnames[idx])
        im = cv2.resize(im, (224, 224))
        im = cv2.cvtColor(im, eval(f"cv2.COLOR_BGR2{self.colormode.upper()}"))
        im = im[66:66+96, 60:60+96, :]
        im = im[self.top_patch[0]:self.top_patch[0]+self.im_size, self.top_patch[1]:self.top_patch[1]+self.im_size, :]
        imgs = self.transform(im)
        label = self.fnames[idx].split("/")[-2]
        if label == "spoof" in label or "device" in label or "print" in label or "video-replay" in label:
#             label = torch.ones(imgs.size(0))
            label = torch.tensor(1)
        elif label == "live" or label == "real":
#             label = torch.zeros(imgs.size(0))
            label = torch.tensor(0)
        return imgs, label
    
    
def load_model(modeltype, im_size):
    model = PatchModel(im_size)
    model = torch.nn.DataParallel(model, device_ids)
    model_path = f"../ckpts/patch_based_cnn/{im_size}/model.pth"
    state_dict = torch.load(model_path)
#     print(state_dict)
    model.load_state_dict(state_dict)
#     print(model)
    model.cuda(device_ids[0])
    model.eval()
    
    return model
    
def infer(imgs, im_path, model):
    imgs = imgs.cuda(device_ids[0])
    outputs = model(imgs)
    return outputs


def load_dataset(im_size, dtype):
    dataset = ScoreDataset(path[dtype], transform=transform, im_size=im_size, modeltype="patch", top_patch=top_patches[im_size])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=64)
    return dataloader



def get_results(im_size, dtype):
    
    model = load_model(modeltype, im_size)
    dataloader = load_dataset(im_size, dtype)
    for idx, (img, im_path, label) in enumerate(dataloader):
#         print(im_path)
        outs = infer(img, im_path, model)
        _, preds = torch.max(outs, 1)
        preds = preds.data.detach().cpu().numpy()
        label = label.data.numpy()
#         print(preds.shape, preds.shape[0], label.shape)
#         print(preds[0], label[0])
        for num in range(preds.shape[0]):
            if preds[num] == label[num]:
                correct[dtype][im_size][str(preds[num])].append(im_path[num])
            else:
                wrong[dtype][im_size][str(preds[num])].append(im_path[num])
        if idx % 10 == 0:
            print(f"{(idx + 1)*batch_size} images are done!!!")
        if len(correct[dtype][im_size]) == 5 and len(wrong[dtype][im_size]):
            return
    return

def select_five(lis):
#     print(lis)
    
    if len(lis) > 5:
        subset = sample([i for i in range(len(lis))], 5)
#         lis = np.random.choice(lis, 5)
        lis = np.array(lis)[subset]
    elif len(lis) < 5:
        none_lis = [None] * (5 - len(lis))
        lis.extend(none_lis)
    return lis

def get_imdata(path, im_size):
    im = cv2.imread(path)
    if im is not None:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (224, 224))
        im = im[66:66+96, 60:60+96, :]
        print(top_patches[im_size])
        patch = top_patches[im_size]
        im = im[patch[0]:patch[0]+im_size, patch[1]:patch[1]+im_size, :]
    else:
        im = np.zeros((im_size, im_size, 3))
#     plt.imshow(patch_grid[8574][24:24+48, 24:24+48, :])
#     plt.show()
#     batches = get_batch_imgs(patch_grid, [top_msu[0]])
#     print(batches.shape)
    
    return im

def get_wrong_and_corrects():
    for dtype in datatypes:
        wrong[dtype] = {}
        correct[dtype] = {}
        
        for im_size in im_sizes:
            wrong[dtype][im_size] = {"0": [], "1": []}
            correct[dtype][im_size] = {"0": [], "1": []}
            
            get_results(im_size, dtype)
            print(f"Done with {im_size} patches")
            wrong[dtype][im_size]["0"], correct[dtype][im_size]["0"]  = select_five(wrong[dtype][im_size]["0"]), select_five(correct[dtype][im_size]["0"])
            wrong[dtype][im_size]["1"], correct[dtype][im_size]["1"]  = select_five(wrong[dtype][im_size]["1"]), select_five(correct[dtype][im_size]["1"])
    
    return {"c": correct, "w": wrong}

def plot_figures(figures, im_key):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
#     figures = {f"{title[j]}-{str(i)}": get_imdata(lis[i]) for i in range(len(lis[j])) for j in range(len(lis))}
    nrows = 8
    
    ncols = 5
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, 20))
    for ind,title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.jet())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    plt.savefig(f'images/{im_key}_patch_info.png')
    plt.show()

label_data = get_wrong_and_corrects()
figures = {24: {}, 48:{}}
label_dict = {"0": "Real", "1": "Spoof"}
for error_key in label_data.keys():
#     print(label_data.keys())
    for data_key in label_data[error_key].keys():
#         print(label_data[error_key].keys())
        for im_key in label_data[error_key][data_key].keys():
#             print(label_data[error_key][data_key].keys())
            for label_key in label_data[error_key][data_key][im_key].keys():
#                 print(label_data[error_key][data_key][im_key].keys())
                for i in range(5):
                    image_path = label_data[error_key][data_key][im_key][label_key][i]
#                     print(image_path)
                    im = get_imdata(image_path, im_key)
                    print(im.shape)
                    figures[im_key][f"{error_key}-{data_key}-{label_dict[label_key]}-{im_key}-{i}"] = im
                    
                    
for im_key in figures.keys():
    plot_figures(figures[im_key], im_key)