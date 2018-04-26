import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
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
        self.v = 0
        self.ev = 0

    def __call__(self, cur_value, n, epoch):
        if self.cur_epoch != epoch:
            self.n = 0
            self.v = 0
            self.ev = 0
            self.cur_epoch = epoch
        self.n = self.n + n
        self.v = self.v + cur_value
        self.ev = self.v / self.n
        return self.ev


def topk_err_sum(pred, targ, k):
    bs = pred.size(0)
    pred_classes = pred.topk(k, 1)[1]
    targ_expanded = targ.view(-1, 1).expand(bs, k)
    pred_err = (pred_classes == targ_expanded).sum(1) == 0
    topk_err_sum = pred_err.sum().item()
    return topk_err_sum


class CrossEntropyCriterion():
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
        self.loss_logger = ValueLogger()
        self.top1_logger = ValueLogger()

    def __call__(self, prediction, target, epoch):
        target = target.view(-1)
        self.loss = self.criterion(prediction, target.view(-1))
        top1_err_sum = topk_err_sum(prediction.detach(), target.detach(), 1)
        bs = prediction.size(0)

        self.stats = {
            'Cross Entropy Loss': self.loss_logger(self.loss.item() * bs, bs, epoch),
            'Top 1 Error': self.top1_logger(top1_err_sum, bs, epoch),
        }
        return self.loss, self.stats


# networks

class BottleneckResmap(nn.Module):

    def __init__(self, ni, no, stride, padding):
        super(BottleneckResmap, self).__init__()

        self.bn1 = nn.BatchNorm2d(ni)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(ni, no // 4, 1, 1)

        self.bn2 = nn.BatchNorm2d(no // 4)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(no // 4, no // 4, 3, stride, padding)

        self.bn3 = nn.BatchNorm2d(no // 4)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv2d(no // 4, no, 1, 1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv3(x)

        return x


class BottleneckBlock(BottleneckResmap):

    def __init__(self, nf, stride):
        if stride == 1:
            super(BottleneckBlock, self).__init__(nf, nf, 1, 1)
            self.downsample = lambda x: x
        elif stride == 2:
            super(BottleneckBlock, self).__init__(nf, nf * 2, 2, 1)
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(nf),
                nn.ReLU(),
                nn.Conv2d(nf, nf * 2, 1, 2)
            )
        else:
            raise Exception("Invalid stride value: {}".format(stride))

        self.resmap = super(BottleneckBlock, self).forward

    def forward(self, x):
        return self.resmap(x) + self.downsample(x)


class ResNet(nn.Module):
    def __init__(self, n_classes, nf_init, n_downsample, n_inter_block):
        super(ResNet, self).__init__()

        nf = nf_init
        features = [nn.Conv2d(3, nf, 3, 1, 1)]
        for i in range(n_downsample):
            for j in range(n_inter_block):
                features.append(BottleneckBlock(nf, 1))
            features.append(BottleneckBlock(nf, 2))
            nf = nf * 2

        features.append(nn.BatchNorm2d(nf))
        features.append(nn.ReLU())
        
        self.features = nn.Sequential(*features)

        self.classifier = nn.Linear(nf, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


# utils

def load_network(network_name, network_args):
    # instantiate
    if network_name == 'resnet':
        network = ResNet(*network_args)
    else:
        raise Exception("Invalid network name: {}".format(network_name))

    # initialize parameters
    for m in network.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.detach())
            if m.bias is not None:
                init.constant_(m.bias.detach(), 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight.detach(), 1)
            init.constant_(m.bias.detach(), 0)
        elif isinstance(m, nn.Linear):
            if m.bias is not None:
                init.constant_(m.bias.detach(), 0)

    # use GPU is available
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
        network_args,
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
    network = load_network(network_name, network_args)

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

