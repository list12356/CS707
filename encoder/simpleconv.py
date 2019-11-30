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

class SimpleConv2(BaseEncoder):
    def __init__(self, num_code = 3, num_hidden=128, img_size=[3, 32, 32]):
        super(SimpleConv2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_hidden//4, kernel_size=3, stride=2, padding=1),  # b, 32, 16, 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # b, 64, 9, 9

            #bottleneck 1
            nn.Conv2d(num_hidden//4, num_hidden//4, 1, stride=1, padding=0),  # b, 32, 9, 9
            nn.ReLU(True),
            nn.Conv2d(num_hidden//4, num_hidden//4, 3, stride=1, padding=1),  # b, 64, 9, 9
            nn.ReLU(True),
            nn.Conv2d(num_hidden//4, num_hidden//2, 1, stride=1, padding=0),  # b, 64, 9, 9
            nn.ReLU(True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # b, 64, 5, 5
            #bottleneck 2
            nn.Conv2d(num_hidden//2, num_hidden//2, 1, stride=1, padding=0),  # b, 64, 5, 5
            nn.ReLU(True),
            nn.Conv2d(num_hidden//2, num_hidden//2, 3, stride=1, padding=1),  # b, 64, 5, 5
            nn.ReLU(True),
            nn.Conv2d(num_hidden//2, num_hidden, 1, stride=1, padding=0),  # b, 128, 5, 5
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # b, 128, 3, 3
            nn.Conv2d(num_hidden, num_hidden*2, 3, stride=1, padding=0),  # b, 256, 1, 1
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_hidden*2, num_hidden, 3, stride=1), # b, 128, 3, 3
            nn.ReLU(True),
            #bottleneck
            nn.ConvTranspose2d(num_hidden, num_hidden, 1, stride=1), # b, 128, 3, 3
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hidden, num_hidden, 3, stride=1),  # b, 128, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hidden, num_hidden//2, 1, stride=1), # b, 64, 5, 5
            nn.ReLU(True),

            #bottleneck
            nn.ConvTranspose2d(num_hidden//2, num_hidden//2, 1, stride=1), # b, 64, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hidden//2, num_hidden//2, 3, stride=2, padding=1),  # b, 64, 9, 9
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hidden//2, num_hidden//4, 1, stride=1), # b, 32, 9, 9
            nn.ReLU(True),

            # nn.ConvTranspose2d(num_hidden//2, num_hidden//2, 3, stride=1, padding=1),  # b, 64, 9, 9
            # nn.ReLU(True),

            nn.ConvTranspose2d(num_hidden//4, num_hidden//4, 3, stride=2, padding=1),  # b, 64, 17, 17
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hidden//4, 3, 4, stride=2, padding=2),  # b, 3, 32, 32
            # nn.Sigmoid()
            nn.Tanh()
        )
