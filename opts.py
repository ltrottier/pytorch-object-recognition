import os
import json
from argparse import ArgumentParser

# parse arguments
parser = ArgumentParser()

# dataset
parser.add_argument('--dataset-name', default='cifar10')
parser.add_argument('--dataset-n-classes', type=int, default=10)
parser.add_argument('--dataset-dir', default='datasets/cifar10')
parser.add_argument('--dataset-train-test-ratio', type=float, default=-1)

# dataloader
parser.add_argument('--dataloader-batch-size', type=int, default=32)
parser.add_argument('--dataloader-shuffle', dest='dataloader_shuffle', action='store_true')
parser.add_argument('--no-dataloader-shuffle', dest='dataloader_shuffle', action='store_false')
parser.set_defaults(dataloader_shuffle=True)
parser.add_argument('--dataloader-num-workers', type=int, default=4)
parser.add_argument('--dataloader-drop-last', dest='dataloader_drop_last', action='store_true')
parser.add_argument('--no-dataloader-drop-last', dest='dataloader_drop_last', action='store_false')
parser.set_defaults(dataloader_drop_last=True)
parser.add_argument('--dataloader-augment', dest='dataloader_augment', action='store_true')
parser.add_argument('--no-dataloader-augment', dest='dataloader_augment', action='store_false')
parser.set_defaults(dataloader_augment=True)

# experiment
parser.add_argument('--experiment-folder', default='results/exp1')

# optim
parser.add_argument('--optim-type', default='sgd')
parser.add_argument('--optim-n-epoch', type=int, default=300)
parser.add_argument('--optim-lr-init', type=float, default=0.1)
parser.add_argument('--optim-lr-schedule', nargs='+', default=[100, 180, 240, 280], type=int)
parser.add_argument('--optim-lr-decay', type=float, default=0.2)
parser.add_argument('--optim-momentum', type=float, default=0.9)
parser.add_argument('--optim-nesterov', dest='optim_nesterov', action='store_true')
parser.add_argument('--no-optim-nesterov', dest='optim_nesterov', action='store_false')
parser.set_defaults(optim_nesterov=True)
parser.add_argument('--optim-weight-decay', type=float, default=5e-4)

# network
parser.add_argument('--network-name', default='resnet')
parser.add_argument('--network-args', nargs='+', default=[10, 32, 2, 5], type=int)

# criterion
parser.add_argument('--criterion-train', default='cross_entropy')
parser.add_argument('--criterion-test', default='cross_entropy')

args = parser.parse_args()
opts = vars(args)

# create experiment folder
experiment_folder = opts['experiment_folder']
os.makedirs(experiment_folder)

# save opts
opts_filename = os.path.join(experiment_folder, 'opts.txt')
with open(opts_filename, 'w') as fid:
    json.dump(opts, fid)

