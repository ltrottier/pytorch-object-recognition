import matplotlib as mpl
mpl.use('Agg')
import os
import dataset
import models
import train
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('optsfile', help='opts file generated with opts.py')
args = parser.parse_args()

# load opts
with open(args.optsfile, 'r') as fid:
    opts = json.load(fid)

# initialize dataset
dataset_name = opts['dataset_name']
dataset_dir = opts['dataset_dir']
train_test_ratio = opts['dataset_train_test_ratio']
batch_size = opts['dataloader_batch_size']
shuffle = opts['dataloader_shuffle']
num_workers = opts['dataloader_num_workers']
drop_last = opts['dataloader_drop_last']

dataset_init = dataset.initialize(
    dataset_name,
    dataset_dir,
    train_test_ratio,
    batch_size,
    shuffle,
    num_workers,
    drop_last)
dataset_train, dataset_test = dataset_init[0]
dataloader_train, dataloader_test = dataset_init[1]

# initialize model
experiment_folder = opts['experiment_folder']
network_name = opts['network_name']
network_args = opts['network_args']
optimizer_type = opts['optim_type']
lr_init = opts['optim_lr_init']
lr_decay = opts['optim_lr_decay']
lr_schedule = opts['optim_lr_schedule']
momentum = opts['optim_momentum']
weight_decay = opts['optim_weight_decay']
nesterov = opts['optim_nesterov']
criterion_name_train = opts['criterion_train']
criterion_name_test = opts['criterion_test']

model_init = models.initialize(
    experiment_folder,
    network_name,
    network_args,
    criterion_name_train,
    criterion_name_test,
    optimizer_type,
    lr_init,
    lr_decay,
    lr_schedule,
    momentum,
    weight_decay,
    nesterov)
network = model_init[0]
optimizer = model_init[1]
callbacks_train, callbacks_test = model_init[2]
criterion_train, criterion_test = model_init[3]

# train
n_epoch = opts['optim_n_epoch']

train.loop(network, optimizer,
           criterion_train, criterion_test,
           callbacks_train, callbacks_test,
           dataloader_train, dataloader_test,
           n_epoch, experiment_folder)
