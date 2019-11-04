import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision
from torchvision.utils import save_image
from PIL import Image

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.show()

def imsave(img, filename):
    img = torchvision.utils.make_grid(img, nrow=4, padding=2)
    img = img.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    im = Image.fromarray((img*255).astype(np.uint8))
    im.save(filename)