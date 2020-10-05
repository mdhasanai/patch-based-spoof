import sys
import os
from os import path
from focal_loss import FocalLoss
print(path.dirname(__file__))
sys.path.append(path.join(path.dirname(__file__), '..'))

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from models.deeppix.deeppix_six_channel import DeepPixBiS
from trainers.deeppix_trainer import Trainer
import torch
from dataloaders.deeppixdataloader import get_dataloaders

# Specify other training parameters
batch_size = 32
num_workers = 4
epochs = 50
learning_rate = 0.0001
weight_decay = 0.00001
save_interval = 1
seed = 3
use_gpu = True
output_dir = '/home/ec2-user/SageMaker/gaze-research/ckpts/deeppix/six_channel/proto3/'
save_path = f'{output_dir}/proto3-best.pt'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


network = DeepPixBiS(pretrained=True)
gpus = [i for i in range(4)]
if torch.cuda.device_count() > 1:
    network = nn.DataParallel(network, device_ids=gpus).cuda(gpus[0])
# set trainable parameters
for name, param in network.named_parameters():
    param.requires_grad = True


# loss definitions
criterion_pixel = nn.BCELoss()

criterion_bce = nn.BCELoss()

# optimizer initialization
optimizer = optim.Adam(filter(lambda p: p.requires_grad, network.parameters(
)), lr=learning_rate, weight_decay=weight_decay)


def compute_loss(network, img, labels, device):
    """
    Compute the losses, given the network, data and labels and 
    device in which the computation will be performed. 
    """
 
#     imagesv = Variable(img['image'].to(device))
    labelsv_pixel = Variable(labels['pixel_mask'].to(device))
    labelsv_binary = Variable(labels['binary_target'].to(device))
    out = network(img)

    beta = 0.5
#     loss_pixel = {}
    loss_pixel = criterion_pixel(out[0].squeeze(1), labelsv_pixel.float())
    loss_bce = criterion_bce(out[1], labelsv_binary.unsqueeze(1).float())
    loss = beta*loss_bce + (1.0-beta)*loss_pixel

    return loss


trainer = Trainer(network, optimizer, compute_loss, learning_rate=learning_rate,
                  batch_size=batch_size, device=f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu', do_crossvalidation=True, save_interval=save_interval, save_path=save_path)

print("Data loading started!")
# train_dataloader, val_dataloader = get_dataloaders()
dataloader = get_dataloaders(data_type="oulu", color_mode=['hsv', 'ycr_cb'], hard_protocol='Protocol_3')#rgb
print("Data Loaded!")
trainer.train(dataloader, n_epochs=epochs, output_dir=output_dir)
