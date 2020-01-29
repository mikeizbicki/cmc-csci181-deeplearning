#!/bin/python3

# process command line args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=16)
parser.add_argument('--alpha',type=float,default=0.01)
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--show_image',action='store_true')
args = parser.parse_args()

# imports
import datetime
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
trainset = torchvision.datasets.CIFAR10(
    root = './data',
    train = True,
    download = True,
    transform = transform,
    )
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size = args.batch_size,
    shuffle = True,
    )

testset = torchvision.datasets.CIFAR10(
    root = './data',
    train = True,
    download = True,
    transform = transform,
    )
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size = args.batch_size,
    shuffle = True,
    )

# display images
if args.show_image:
    import matplotlib.pyplot as plt
    import numpy as np

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))

# define the model
w = torch.tensor(torch.rand(3,32,32,10),requires_grad=True)
b = torch.tensor(torch.rand(10),requires_grad=True)

def f(x):
    return torch.einsum('bijk,ijkl -> bl',x,w)+b

# optimize
criterion = nn.CrossEntropyLoss()
loss=float('inf')
for epoch in range(args.epochs):
    for i, data in enumerate(trainloader, 0):
        if i%1000==0:
            print(
                datetime.datetime.now(),
                'epoch=',epoch,
                'i=',i,
                'loss=',loss
                )
        images, labels = data
        outputs = f(images)
        loss = criterion(outputs,labels)
        loss.backward()
        w = w - args.alpha * w.grad
        b = b - args.alpha * b.grad
        w = torch.tensor(w,requires_grad=True)
        b = torch.tensor(b,requires_grad=True)

# eval on test set
correct = 0.0
total = 0.0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = f(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('test set accuracy = ', correct/total)
