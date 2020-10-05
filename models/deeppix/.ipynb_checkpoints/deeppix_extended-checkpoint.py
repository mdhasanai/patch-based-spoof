import torch
from torch import nn
from torchvision import models


class DeepPixBiS(nn.Module):

    def __init__(self, pretrained=True):
        super(DeepPixBiS, self).__init__()
        base_line = models.densenet161(pretrained=pretrained)
        feature_extractor = list(base_line.children())
        self.enc = nn.Sequential(*feature_extractor[0][0:8])
        self.dec = nn.Conv2d(384, 1, kernel_size=1, padding=0)
        self.concat_layer = nn.Conv2d(2, 1, kernel_size=1, padding=0)
        self.linear = nn.Linear(14 * 14, 1)

    def forward(self, x):
        # pass the 224x224 image through the base model
        out = []
        for key, value in x.items():
            enc = self.enc(value)
            dec = self.dec(enc)
            dec = nn.Sigmoid()(dec)
            out.append(dec.view(-1, 14*14))
        out = torch.tensor(out)
        out = torch.cat(out, dim=1)
        out = self.concat_layer(out)
        op = self.linear(out.view(-1, 14*14))
        op = nn.Sigmoid()(op)
        print(op.shape)

        return out, op
    

