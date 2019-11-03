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
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

def random_save(inputs, num, filename):
    index = np.random.choice(inputs.shape[0], num, replace=False)  
    img = torchvision.utils.make_grid(inputs[index])
    img = img.numpy().transpose((1, 2, 0))
    img = std * img + mean
    im = Image.fromarray(img)
    im.save(filename)