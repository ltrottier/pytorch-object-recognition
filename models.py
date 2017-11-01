import os
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# callbacks


class Callbacks():
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def __call__(self, input, output, target, meta, epoch):
        for callback in self.callbacks:
            callback(input, output, target, meta, epoch)


class LRSchedule():
    def __init__(self, optimizer, lr_init, lr_decay, lr_schedule):
        self.optimizer = optimizer
        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.lr_schedule = np.array(lr_schedule)
        self.cur_epoch = -1

    def __call__(self, input, output, target, meta, epoch):
        if self.cur_epoch != epoch:
            self.cur_epoch = epoch
            lr = self.lr_init * \
                self.lr_decay ** (self.lr_schedule <= epoch).sum()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr


class Unfreeze():
    def __init__(self, network, epoch):
        self.network = network
        self.epoch = epoch

    def __call__(self, input, output, target, meta, epoch):
        if self.epoch == epoch:
            self.network.unfreeze()


# criterion

class ValueLogger():
    def __init__(self):
        self.cur_epoch = 0
        self.n = 0
        self.ex = 0

    def __call__(self, cur_value, epoch):
        if self.cur_epoch != epoch:
            self.n = 0
            self.ex = 0
            self.cur_epoch = epoch
        self.ex = (self.ex * self.n + cur_value) / (self.n + 1)
        self.n = self.n + 1
        return self.ex


def topk_err_mean(pred, targ, k):
    bs = pred.size()[0]
    pred_classes = pred.topk(k, 1)[1]
    targ_expanded = targ.view(-1, 1).expand(bs, k)
    pred_err = (pred_classes == targ_expanded).sum(1) == 0
    topk_err_mean = pred_err.sum() / bs
    return topk_err_mean


class CrossEntropyCriterion():
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
        self.loss_logger = ValueLogger()
        self.top1_logger = ValueLogger()

    def __call__(self, prediction, target, epoch):
        target = target.view(-1)
        self.loss = self.criterion(prediction, target.view(-1))
        top1_err_mean = topk_err_mean(prediction.data, target.data, 1)

        self.stats = {
            'Cross Entropy Loss': self.loss_logger(self.loss.data[0], epoch),
            'Top 1 Error': self.top1_logger(top1_err_mean, epoch),
        }
        return self.loss, self.stats


# utils

class DummyNet(nn.Module):
    def __init__(self):
        super(DummyNet, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(32)
        )
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        y = self.layer(x)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y


def load_network(network_name, pretrained):
    if network_name == 'resnet18':
        network = torchvision.models.resnet18(pretrained=pretrained)
    elif network_name == 'dummynet':
        network = DummyNet()
    else:
        raise Exception("Invalid network name: {}".format(network_name))

    if torch.cuda.is_available():
        network = network.cuda()

    return network


def load_train_test_criterions(criterion_name_train, criterion_name_test):

    criterions = []
    for criterion_name in [criterion_name_train, criterion_name_test]:
        if criterion_name == 'cross_entropy':
            criterion = CrossEntropyCriterion()
        else:
            raise Exception(
                "Invalid criterion name: {}".format(criterion_name))

        criterions.append(criterion)

    return tuple(criterions)


def load_optimizer(
        network,
        optimizer_type,
        lr_init,
        momentum,
        weight_decay,
        nesterov):
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(
            network.parameters(),
            lr=lr_init,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(
            network.parameters(),
            lr=lr_init,
            weight_decay=weight_decay)
    else:
        raise Exception("Invalid optimizer type: {}".format(optimizer_type))

    return optimizer


def initialize(
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
        nesterov):

    # network
    network = load_network(network_name, pretrained)

    # save network architecture
    n_parameters = np.sum([p.numel() for p in network.parameters()])
    network_filename = os.path.join(experiment_folder, 'network.txt')
    if not os.path.isfile(network_filename):
        with open(network_filename, 'w') as fid:
            fid.write(str(network) + '\n')
            fid.write("Network has {} parameters.\n".format(n_parameters))

    # train test criterions
    criterions = load_train_test_criterions(
        criterion_name_train, criterion_name_test)

    # optimizer
    optimizer = load_optimizer(
        network,
        optimizer_type,
        lr_init,
        momentum,
        weight_decay,
        nesterov)

    # callbacks
    callbacks_train = Callbacks([
        LRSchedule(optimizer, lr_init, lr_decay, lr_schedule),
    ])
    callbacks_test = Callbacks([])
    callbacks = callbacks_train, callbacks_test

    return network, optimizer, callbacks, criterions
