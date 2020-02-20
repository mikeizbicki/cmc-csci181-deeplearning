import argparse

# process command line args
parser = argparse.ArgumentParser()

parser_model = parser.add_argument_group('model options')
parser_model.add_argument('--connections',choices=['plain','resnet'],default='resnet')
parser_model.add_argument('--size',type=int,default=20)

parser_opt = parser.add_argument_group('optimization options')
parser_opt.add_argument('--batch_size',type=int,default=16)
parser_opt.add_argument('--learning_rate',type=float,default=0.01)
parser_opt.add_argument('--epochs',type=int,default=10)
parser_opt.add_argument('--warm_start',type=str,default=None)

parser_data = parser.add_argument_group('data options')
parser_data.add_argument('--dataset',choices=['mnist','cifar10'])

parser_debug = parser.add_argument_group('debug options')
parser_debug.add_argument('--show_image',action='store_true')
parser_debug.add_argument('--print_delay',type=int,default=60)
parser_debug.add_argument('--log_dir',type=str)
parser_debug.add_argument('--eval',action='store_true')

args = parser.parse_args()

# load libraries
import datetime
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# load data
if args.dataset=='cifar10':
    image_shape=[3,32,32]

    transform = transforms.Compose(
        [ transforms.RandomHorizontalFlip()
        , transforms.ToTensor()
        , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
        )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
        )

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
        )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
        )

if args.dataset=='mnist':
    image_shape=[1,28,28]

    transform = transforms.Compose(
        [ transforms.RandomHorizontalFlip()
        , transforms.ToTensor()
        , transforms.Normalize((0.5,), (0.5,))
        ])

    trainset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
        )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
        )

    testset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
        )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
        )

# show image
if args.show_image:
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))

# define the model
def conv3x3(channels_in, channels_out):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        channels_in,
        channels_out,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        bias=False,
        dilation=dilation
        )

class ResnetBlock(nn.Module):
    def __init__(
            self,
            channels,
            use_bn = True,
            ):
        super(BasicBlock, self).__init__()
        norm_layer = torch.nn.BatchNorm2d
        self.use_bn = use_bn
        self.conv1 = conv3x3(channels, channels, stride)
        if self.use_bn:
            self.bn1 = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        if self.use_bn:
            self.bn2 = norm_layer(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

import functools
image_size = functools.reduce(lambda x, y: x * y, image_shape, 1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = torch.nn.Linear(image_size,10)
        pass

    def forward(self, x):
        out = x.view(args.batch_size,image_size)
        out = self.fc(out)
        return out

net = Net()

# load pretrained model
if args.warm_start is not None:
    print('warm starting model from',args.warm_start)
    model_dict = torch.load(os.path.join(args.warm_start,'model'))
    net.load_state_dict(model_dict['model_state_dict'])

# create save dir
log_dir = args.log_dir
if log_dir is None:
    log_dir = 'log/'+str(datetime.datetime.now())

try:
    os.mkdir(log_dir)
except FileExistsError:
    print('cannot create log dir,',log_dir,'already exists')
    sys.exit(1)

writer = SummaryWriter(log_dir=log_dir)

# train the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)
net.train()

total_iter = 0
last_print = 0

steps = 0
for epoch in range(args.epochs):
    for i, data in enumerate(trainloader):
        steps += 1
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # accuracy
        prediction = torch.argmax(outputs,dim=1)
        accuracy = (prediction==labels).float().mean()

        # tensorboard
        writer.add_scalar('train/loss', loss.item(), steps)
        writer.add_scalar('train/accuracy', accuracy.item(), steps)

        # print statistics
        total_iter += 1
        if time.time() - last_print > args.print_delay:
            print(datetime.datetime.now(),'epoch = ',epoch,'steps=',steps,'batch/sec=',total_iter/args.print_delay)
            total_iter = 0
            last_print = time.time()

    torch.save({
            'epoch':epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':loss
        }, os.path.join(log_dir,'model'))


# test set
if args.eval:
    print('evaluating model')
    net.eval()

    loss_total = 0
    accuracy_total = 0
    for i, data in enumerate(testloader):
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # accuracy
        prediction = torch.argmax(outputs,dim=1)
        accuracy = (prediction==labels).float().mean()

        # update variables
        loss_total += loss.item()
        accuracy_total += accuracy.item()

    print('loss=',loss_total/i)
    print('accuracy=',accuracy_total/i)

