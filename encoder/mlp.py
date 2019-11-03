from torch import nn

from encoder.base import BaseEncoder

class MLPEncoder(BaseEncoder):
    def __init__(self, num_code = 3, num_hidden=128, img_size=[3, 32, 32]):
        super(MLPEncoder, self).__init__()
        self.img_size = img_size
        dim = 1 
        for size in img_size:
            dim = dim*size
        self.encoder = nn.Sequential(
            nn.Linear(dim, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_hidden//2),
            nn.ReLU(True),
            nn.Linear(num_hidden//2, num_code)
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_code, num_hidden//2),
            nn.ReLU(True),
            nn.Linear(num_hidden//2, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, dim), 
            nn.Tanh()
        )
    
    def encode(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, x):
        x =  self.decoder(x)
        x = x.view(x.size(0), *self.img_size)
        return x