import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from encoder.mlp import MLPEncoder
from utils import random_save

# classifier = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)

def train(batch_size=64, lr=0.001, num_epoch=10, print_every=100, mode='no_backbone', 
            backbone='mobilenet_v2'):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    encoder = MLPEncoder(num_code=128, num_hidden=512, img_size=[3, 32, 32])
    backbone = getattr(models, backbone)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        encoder.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(num_epoch):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs: N x C x H x W
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # train the encoder
            outputs = encoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            # test on backbone
            # backbone()
            backbone_outs = backbone(inputs)
            _, predicted = torch.max(backbone_outs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item()
            if i % print_every == print_every - 1:
                print('[{:d}, {:5d}] loss: {:.3f}'.format(
                    epoch + 1, i + 1, running_loss / print_every))
                running_loss = 0.0
                correct = 0
                total = 0
                # random show image
                random_save(inputs[:16], 16, "./output/train_original_{!s}.png".format(i))
                random_save(outputs[:16], 16, "./output/train_decode_{!s}.png".format(i))


        torch.save(encoder.state_dict(), './autoencoder.pth')

def validate():
    pass

if __name__ == "__main__":
    train()
