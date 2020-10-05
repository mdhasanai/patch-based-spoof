import torch
from torch import nn
from torchvision import models


class DeepPixBiS(nn.Module):

    def __init__(self, pretrained=True):
        super(DeepPixBiS, self).__init__()
        base_line = models.densenet161(pretrained=pretrained)
        feature_extractor = list(base_line.children())
        self.conv1 = nn.Conv2d(6, 3, 1)
        self.enc = nn.Sequential(*feature_extractor[0][0:8])
        self.dec = nn.Conv2d(384, 1, kernel_size=1, padding=0)
        self.linear = nn.Linear(14 * 14, 1)

    def forward(self, x):
        # pass the 224x224 image through the base model
        concat = []
        for k, v in x.items():
            concat.append(v)
        x = torch.cat(concat, dim=1)
#         print(x.shape)
        x = self.conv1(x)
        enc = self.enc(x)
        # generate a 14x14 map
        dec = self.dec(enc)
        dec = nn.Sigmoid()(dec)

        # final linear output layer
        dec_flat = dec.view(-1, 14 * 14)
        op = self.linear(dec_flat)
        op = nn.Sigmoid()(op)

        return dec, op
    

