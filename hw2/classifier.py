#!/bin/python3
'''
> python3 classifier.py --dataset=mnist --model=linear
test set accuracy =  0.9126833333333333
> python3 classifier.py --dataset=mnist --model=factorized_linear
test set accuracy =  0.8846833333333334
> python3 classifier.py --dataset=mnist --model=neural_network --size=256
test set accuracy =  0.92685
> python3 classifier.py --dataset=mnist --model=kitchen_sink --size=256
test set accuracy =  0.92685
'''

# process command line args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=16)
parser.add_argument('--alpha',type=float,default=0.01)
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--show_image',action='store_true')
parser.add_argument('--size',type=int,default=32)
parser.add_argument('--print_step',type=int,default=1000)
parser.add_argument('--dataset',choices=['mnist','cifar10'])
parser.add_argument('--ema_alpha',type=float,default=0.99)
parser.add_argument('--model',choices=['linear','factorized_linear','kitchen_sink','neural_network'],default='linear')
parser.add_argument('--eval_each_epoch',action='store_true')
args = parser.parse_args()

# imports
import datetime
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# load dataset
if args.dataset=='cifar10':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ],
        )
    trainset = torchvision.datasets.CIFAR10(
        root = './data',
        train = True,
        download = True,
        transform = transform,
        )
    testset = torchvision.datasets.CIFAR10(
        root = './data',
        train = True,
        download = True,
        transform = transform,
        )
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(( 0.5,), ( 0.5,))
        ],
        )
    trainset = torchvision.datasets.MNIST(
        root = './data',
        train = True,
        download = True,
        transform = transform,
        )
    testset = torchvision.datasets.MNIST(
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

    # exit
    import sys
    sys.exit(0)

# define the model
images, labels = iter(trainloader).next()
shape_input = images.shape[1:]
shape_output = torch.Size([10])
h = torch.Size([args.size])

w = torch.tensor(torch.rand(shape_input+shape_output),requires_grad=True,device=device)
u = torch.tensor(torch.rand(shape_input+h),requires_grad=True,device=device)
v = torch.tensor(torch.rand(h+shape_output),requires_grad=True,device=device)

def linear(x):
    return torch.einsum('bijk,ijkl -> bl',x,w)

def factorized_linear(x):
    return torch.einsum('bijk,ijkh,hl -> bl',x,u,v)

relu = nn.ReLU()
def neural_network(x):
    net = torch.einsum('bijk,ijkh -> bh',x,u)
    net = relu(net)
    net = torch.einsum('bh,hl -> bl',net,v)
    return net

kitchen_sink = neural_network

f = eval(args.model)

# eval on test set
def eval_test_set():
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

# optimize
criterion = nn.CrossEntropyLoss()
loss = float('inf')
loss_ave = loss
for epoch in range(args.epochs):
    for i, data in enumerate(trainloader, 0):
        if i%args.print_step==0:
            print(
                datetime.datetime.now(),
                'epoch=',epoch,
                'i=',i,
                'loss_ave=',loss_ave
                )
        images, labels = data
        images.cuda()
        labels.cuda()
        outputs = f(images)
        loss = criterion(outputs,labels)
        if loss_ave == float('inf'):
            loss_ave = loss
        else:
            loss_ave = args.ema_alpha * loss_ave + (1 - args.ema_alpha) * loss
        loss.backward()
        if args.model=='linear':
            w = w - args.alpha * w.grad
            w = torch.tensor(w,requires_grad=True)
        else:
            if args.model!='kitchen_sink':
                u = u - args.alpha * u.grad
                u = torch.tensor(u,requires_grad=True)
            v = v - args.alpha * v.grad
            v = torch.tensor(v,requires_grad=True)

    if args.eval_each_epoch:
        eval_test_set()

if not args.eval_each_epoch:
    eval_test_set()

