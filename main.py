import os
import dataset
import models
import train
import json
from opts import opts

# create experiment folder
experiment_folder = opts['experiment']['folder']
os.makedirs(experiment_folder)

# save opts
opts_filename = os.path.join(experiment_folder, 'opts.txt')
with open(opts_filename, 'w') as fid:
    json.dump(opts, fid)

# initialize dataset
dataset_name = opts['dataset']['name']
dataset_dir = opts['dataset']['dir']
train_test_ratio = opts['dataset']['train_test_ratio']
batch_size = opts['dataloader']['batch_size']
shuffle = opts['dataloader']['shuffle']
num_workers = opts['dataloader']['num_workers']
drop_last = opts['dataloader']['drop_last']

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
network_name = opts['network']['name']
pretrained = opts['network']['pretrained']
optimizer_type = opts['optim']['type']
lr_init = opts['optim']['lr_init']
lr_decay = opts['optim']['lr_decay']
lr_schedule = opts['optim']['lr_schedule']
momentum = opts['optim']['momentum']
weight_decay = opts['optim']['weight_decay']
nesterov = opts['optim']['nesterov']
criterion_name_train = opts['criterion']['train']
criterion_name_test = opts['criterion']['test']

model_init = models.initialize(
    experiment_folder,
    network_name,
    pretrained,
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
n_epoch = opts['optim']['n_epoch']

train.loop(network, optimizer,
           criterion_train, criterion_test,
           callbacks_train, callbacks_test,
           dataloader_train, dataloader_test,
           n_epoch, experiment_folder)
