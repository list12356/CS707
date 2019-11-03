__author__ = 'SherlockLiao'

import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

class BaseEncoder(nn.Module):
    def __init__(self):
        super(BaseEncoder, self).__init__()
        self.encoder = None
        self.decoder = None
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x