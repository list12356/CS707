from torch import nn

from encoder.base import BaseEncoder

class SimpleConv(BaseEncoder):
    def __init__(self, num_code = 3, num_hidden=128, img_size=[3, 32, 32]):
        super(SimpleConv, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_hidden//2, kernel_size=3, stride=2, padding=1),  # b, 64, 16, 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # b, 64, 9, 9
            nn.Conv2d(num_hidden//2, num_hidden, 3, stride=2, padding=1),  # b, 128, 5, 5
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # b, 128, 3, 3
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_hidden, num_hidden//2, 3, stride=2),  # b, 64, 7, 7
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hidden//2, num_hidden//2, 3, stride=2),  # b, 64, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hidden//2, 3, 4, stride=2),  # b, 3, 32, 32
            nn.Tanh()
        )