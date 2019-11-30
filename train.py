import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os

from encoder.mlp import MLPEncoder
from encoder.simpleconv import SimpleConv, SimpleConv2
from utils import imsave

# classifier = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)

def train(batch_size=500, lr=0.001, num_epoch=100, print_every=100, mode='no_backbone', 
            backbone='simplenet', encoder='simpleconv', dataset="cifar10", resume=None,
            train=True, bb_coeff=1, save_path=None):
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    log = open('./train.log.' + str(datetime.datetime.now()), 'w+')
    if os.path.exists('./saved_models') == False:
        os.makedirs('./saved_models')
    if os.path.exists('./output') == False:
        os.makedirs('./output')
    if save_path is None:
        save_path = './saved_models/{!s}_encoder_{!s}.pth'.format(encoder, bb_coeff)
    transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform)
    elif dataset == "imagenet":
        trainset = torchvision.datasets.ImageNet(root='./data', split="train",
                                                download=True, transform=transform)
        testset = torchvision.datasets.ImageNet(root='./data', split="val",
                                            download=True, transform=transform)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=16, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=16, pin_memory=True)

    if encoder == 'mlp':
        encoder = MLPEncoder(num_code=256, num_hidden=1024, img_size=[3, 32, 32])
        encoder.to(device)
    elif encoder == 'simpleconv':
        encoder = SimpleConv(num_code=256, num_hidden=256, img_size=[3, 32, 32])
        encoder.to(device)
    elif encoder == 'simpleconv2':
        encoder = SimpleConv2(num_code=256, num_hidden=128, img_size=[3, 32, 32])
        encoder.to(device)
    else:
        print('unknown encoder')
        return
    
    if backbone == "simplenet":
        from backbone.simplenet import SimpleNet
        backbone = SimpleNet()
        backbone.load_state_dict(torch.load('./saved_models/simplenet.pth'))
    elif backbone == "res56":
        import backbone.resnet as resnet
        backbone = resnet.__dict__["resnet56"]()
        backbone.load_state_dict(torch.load('./saved_models/res56-cifar10.pth'))
    else:
        backbone = getattr(models, backbone)(pretrained=True)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.to(device)

    if train == True:
        criterion = nn.MSELoss()
        backbone_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            encoder.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        # optimizer = torch.optim.Adam(
        #     encoder.parameters(), lr=lr, weight_decay=1e-5)
        best_acc = 0
        start_epoch = 0
        if resume is not None:
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            encoder.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim'])
            best_acc = checkpoint['best_acc']

        for epoch in range(start_epoch, start_epoch+num_epoch):
            running_loss = 0.0
            correct_input = 0
            correct_output = 0
            total = 0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                # inputs: N x C x H x W
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # train the encoder
                outputs = encoder(inputs)
                backbone_outs = backbone(outputs)
                if mode == "backbone":
                    loss = (1 - bb_coeff)*criterion(outputs, inputs) + bb_coeff*backbone_criterion(backbone_outs, labels)
                else:
                    loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                # test on backbone
                with torch.no_grad():
                    total += labels.size(0)
                    backbone_outs = backbone(inputs)
                    _, predicted = torch.max(backbone_outs.data, 1)
                    correct_input += (predicted == labels).sum().item()
                    backbone_outs = backbone(outputs)
                    _, predicted = torch.max(backbone_outs.data, 1)
                    correct_output += (predicted == labels).sum().item()
                # total = 1

                # print statistics
                running_loss += loss.item()
                if i % print_every == print_every - 1:
                    print('[{:d}, {:5d}] loss: {:.3f} Acc on input: {:.3f} Acc on output: {:.3f}'.format(
                        epoch + 1, i + 1, running_loss / i, correct_input/total, correct_output/total))
                    correct_input = 0
                    correct_output = 0
                    total = 0
                    # random show image
                    index = np.random.choice(inputs.shape[0], 16, replace=False)  
                    imsave(inputs[index].cpu(), "./output/train_original_{!s}.png".format(i))
                    imsave(outputs[index].detach().cpu(), "./output/train_decode_{!s}.png".format(i))
            
            acc = test(testloader, device, print_every, encoder, backbone)
            log.write('{:d}, {:.3f}, {:.3f}'.format(epoch + 1, running_loss/i, acc))
            log.flush()
            if best_acc > acc: 
                checkpoint = {
                    "state_dict": encoder.state_dict(),
                    "best_acc": best_acc,
                }
                torch.save(checkpoint, save_path)
                best_acc = acc
            elif epoch % 5 == 0:
                checkpoint = {
                    "state_dict": encoder.state_dict(),
                    "optim": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "epoch": epoch
                }
                torch.save(checkpoint, save_path + str(epoch))


    test(testloader, device, print_every, encoder, backbone)

def test(testloader, device, print_every, encoder, backbone):
    correct_input = 0
    correct_output = 0
    total = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = encoder(inputs)

        # test on backbone
        total += labels.size(0)
        backbone_outs = backbone(inputs)
        _, predicted = torch.max(backbone_outs.data, 1)
        correct_input += (predicted == labels).sum().item()
        backbone_outs = backbone(outputs)
        _, predicted = torch.max(backbone_outs.data, 1)
        correct_output += (predicted == labels).sum().item()

        if i % 20 == 0:
            # random show image
            index = np.random.choice(inputs.shape[0], 16, replace=False)  
            imsave(inputs[index].cpu(), "./output/test_original_{!s}.png".format(i))
            imsave(outputs[index].detach().cpu(), "./output/test_decode_{!s}.png".format(i))

    print('Acc on input: {:.3f} Acc on output: {:.3f}'.format(
        correct_input/total, correct_output/total))
    return correct_output/total

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='no_backbone')
    parser.add_argument('--backbone', default='res56')
    parser.add_argument('--bb', default=1.0, type=float)
    parser.add_argument('--encoder', default='simpleconv')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--resume')
    parser.add_argument('--save_path')
    args = parser.parse_args()
    args.train = True if args.train == 1 else False
    train(num_epoch=args.num_epoch, mode=args.mode, backbone=args.backbone, \
            train=args.train, encoder=args.encoder, dataset=args.dataset, \
            resume=args.resume, lr=args.lr, batch_size=args.batch_size,\
            save_path=args.save_path, bb_coeff=args.bb)
