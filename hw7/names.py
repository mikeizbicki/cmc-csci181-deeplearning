#!/usr/bin/env python3

# process command line args
import argparse
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser_control = parser.add_argument_group('control options')
parser_control.add_argument('--infer',action='store_true')
parser_control.add_argument('--train',action='store_true')

parser_data = parser.add_argument_group('data options')
parser_data.add_argument('--data',default='names')

parser_model = parser.add_argument_group('model options')
parser_model.add_argument('--model',choices=['cnn','rnn','gru','lstm'],default='rnn')
parser_model.add_argument('--hidden_layer_size',type=int,default=128)
parser_model.add_argument('--num_layers',type=int,default=1)

parser_opt = parser.add_argument_group('optimization options')
parser_opt.add_argument('--batch_size',type=int,default=1)
parser_opt.add_argument('--learning_rate',type=float,default=1e-1)
parser_opt.add_argument('--optimizer',choices=['sgd','adam'],default='sgd')
parser_opt.add_argument('--gradient_clipping',action='store_true')
parser_opt.add_argument('--momentum',type=float,default=0.9)
parser_opt.add_argument('--weight_decay',type=float,default=1e-4)
parser_opt.add_argument('--samples',type=int,default=10000)
parser_opt.add_argument('--input_length',type=int,default=20)
parser_opt.add_argument('--warm_start')

parser_debug = parser.add_argument_group('debug options')
parser_debug.add_argument('--print_delay',type=int,default=5)
parser_debug.add_argument('--log_dir',type=str)
parser_debug.add_argument('--save_every',type=int,default=1000)

args = parser.parse_args()

if args.model=='cnn' and args.input_length is None:
    raise ValueError('if --model=cnn, then you must specify --input_length')

# load args from file if warm starting
if args.warm_start is not None:
    import sys
    import os
    args_orig = args
    args = parser.parse_args(['@'+os.path.join(args.warm_start,'args')]+sys.argv[1:])
    args.train = args_orig.train

# supress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# load modules
import datetime
import glob
import os
import math
import random
import string
import sys
import time
import unicodedata

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# import the training data
vocabulary = string.ascii_letters + " .,;'$"

def unicode_to_ascii(s):
    '''
    Removes diacritics from unicode characters.
    See: https://stackoverflow.com/a/518232/2809427
    '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in vocabulary
    )

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []
for filename in glob.glob(os.path.join(args.data,'*.txt')):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    lines = [unicode_to_ascii(line) for line in lines]
    category_lines[category] = lines

def str_to_tensor(ss,input_length=None):
    '''
    Converts a list of strings into a tensor of shape <max_length, len(ss), len(vocabulary)>.
    This is used to convert text into a form suitable for input into a RNN/CNN.
    '''
    max_length = max([len(s) for s in ss]) + 1
    if input_length:
        max_length = input_length
    tensor = torch.zeros(max_length, len(ss), len(vocabulary))
    for j,s in enumerate(ss):
        s+='$'
        for i, letter in enumerate(s):
            if i<max_length:
                tensor[i,j,vocabulary.find(letter)] = 1
    return tensor

# define the model
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel,self).__init__()
        if args.model=='rnn':
            self.rnn = nn.RNN(len(vocabulary),args.hidden_layer_size,num_layers=args.num_layers)
        if args.model=='gru':
            self.rnn = nn.GRU(len(vocabulary),args.hidden_layer_size,num_layers=args.num_layers)
        if args.model=='lstm':
            self.rnn = nn.LSTM(len(vocabulary),args.hidden_layer_size,num_layers=args.num_layers)
        self.fc = nn.Linear(args.hidden_layer_size,len(all_categories))

    def forward(self, x):
        out,h_n = self.rnn(x)
        out = self.fc(out[out.shape[0]-1,:,:])
        return out

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel,self).__init__()
        self.relu = nn.ReLU()
        self.cnn = \
            nn.Conv1d(len(vocabulary),args.hidden_layer_size,3,padding=1)
        self.cnns = (args.num_layers-1)*[
            nn.Conv1d(args.hidden_layer_size,args.hidden_layer_size,3,padding=1)
            ]
        self.fc = nn.Linear(args.hidden_layer_size*args.input_length,len(all_categories))

    def forward(self,x):
        out = torch.einsum('lbv->bvl',x)
        out = self.cnn(out)
        out = self.relu(out)
        for cnn in self.cnns:
            out = cnn(out)
            out = self.relu(out)
        out = out.view(args.batch_size,args.hidden_layer_size*args.input_length)
        out = self.fc(out)
        return out

# load the model
if args.model=='cnn':
    model = CNNModel()
else:
    model = RNNModel()

if args.warm_start:
    print('warm starting model from',args.warm_start)
    model_dict = torch.load(os.path.join(args.warm_start,'model'))
    model.load_state_dict(model_dict['model_state_dict'])

# training
if args.train:

    # create log_dir
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = 'log/'+(
            'model='+args.model+
            '_lr='+str(args.learning_rate)+
            '_optim='+args.optimizer+
            '_clip='+str(args.gradient_clipping)+
            '_'+str(datetime.datetime.now())
            )
    try:
        os.makedirs(log_dir)
        with open(os.path.join(log_dir,'args'), 'w') as f:
            f.write('\n'.join(sys.argv[1:]))
    except FileExistsError:
        print('cannot create log dir,',log_dir,'already exists')
        sys.exit(1)
    writer = SummaryWriter(log_dir=log_dir)

    # prepare model for training
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay
                )
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
                )
    model.train()

    # training loop
    start_time = time.time()
    for step in range(1, args.samples + 1):

        # get random training example
        categories = []
        lines = []
        for i in range(args.batch_size):
            category = random.choice(all_categories)
            line = random.choice(category_lines[category])
            categories.append(all_categories.index(category))
            lines.append(line)
        category_tensor = torch.tensor(categories, dtype=torch.long)
        line_tensor = str_to_tensor(lines,args.input_length)

        # perform training step
        output = model(line_tensor)
        loss = criterion(output, category_tensor)
        loss.backward()
        grad_norm = sum([ torch.norm(p.grad)**2 for p in model.parameters()])**(1/2)
        if args.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()

        # get category from output
        top_n, top_i = output.topk(1)
        guess_i = top_i[-1].item()
        category_i = category_tensor[-1]
        guess = all_categories[guess_i]
        category = all_categories[category_i]
        accuracies = torch.where(
                top_i[:,0]==category_tensor,
                torch.ones([args.batch_size]),
                torch.zeros([args.batch_size])
                )
        accuracy = torch.mean(accuracies).item()

        # tensorboard
        writer.add_scalar('train/loss', loss.item(), step)
        writer.add_scalar('train/accuracy', accuracy, step)
        writer.add_scalar('train/grad_norm', grad_norm.item(), step)

        # print status update
        if step % 100 == 0:
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%.2f sec) %.4f %s / %s %s' % (
                step,
                step / args.samples * 100,
                time.time()-start_time,
                loss,
                line,
                guess,
                correct
                ))

        # save model
        if step%args.save_every == 0 or step==args.samples:
            print('saving model checkpoint')
            torch.save({
                    'step':step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss':loss
                }, os.path.join(log_dir,'model'))


# infer
model.eval()
softmax = torch.nn.Softmax(dim=1)
if args.infer:
    for line in sys.stdin:
        line = line.strip()
        line_tensor = str_to_tensor([line],args.input_length)
        output = model(line_tensor)
        probs = softmax(output)
        top_n, top_i = probs.topk(3)
        guess_0 = all_categories[top_i[0,0].item()]
        guess_1 = all_categories[top_i[0,1].item()]
        guess_2 = all_categories[top_i[0,2].item()]
        print(
            'name=',line,
            'guess0=%s (%0.2f)'%(guess_0,probs[0,top_i[0,0].item()],),
            'guess1=%s (%0.2f)'%(guess_1,probs[0,top_i[0,1].item()],),
            'guess2=%s (%0.2f)'%(guess_2,probs[0,top_i[0,2].item()],),
            )

