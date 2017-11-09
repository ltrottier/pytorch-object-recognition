import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import json
import plot

# logger


class JSONLogger():
    def __init__(self, filename):
        self.filename = filename
        self.values = []
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    def __call__(self, value):
        self.values.append(value)
        with open(self.filename, 'w') as fd:
            json.dump(self.values, fd)


class Timer():
    def __init__(self):
        self.t0 = -1

    def start(self):
        self.t0 = time.time()

    def tick(self):
        t1 = time.time()
        delta = t1 - self.t0
        self.t0 = t1
        return delta


# utils


def make_variable(input):
    if isinstance(input, list):
        output = [Variable(t) for t in input]
    else:
        output = Variable(input)
    return output


def make_gpu_tensor(input):
    if isinstance(input, list):
        output = [t.cuda() for t in input]
    else:
        output = input.cuda()
    return output


def process_array(input):
    output = input
    if torch.cuda.is_available():
        output = make_gpu_tensor(output)
    output = make_variable(output)
    return output


def save_model(model, savepath):
    parameters_path = os.path.join(savepath, 'parameters.pt')
    model_path = os.path.join(savepath, 'model.pt')
    torch.save(model.state_dict(), parameters_path)


# learn

def train_step(network, criterion, optimizer, dataloader, epoch, callbacks):
    stats = None
    network.train(True)
    for i, (input, target, meta) in enumerate(dataloader):
        input = process_array(input)
        target = process_array(target)
        optimizer.zero_grad()
        output = network(input)
        loss, stats = criterion(output, target, epoch)
        callbacks(input, output, target, meta, epoch)
        loss.backward()
        optimizer.step()

    return {'Train Stats': stats}


def test_step(network, criterion, dataloader, epoch, callbacks):
    stats = None
    network.train(False)
    for i, (input, target, meta) in enumerate(dataloader):
        input = process_array(input)
        target = process_array(target)
        output = network(input)
        loss, stats = criterion(output, target, epoch)
        callbacks(input, output, target, meta, epoch)

    return {'Test Stats': stats}


def loop(network, optimizer,
         criterion_train, criterion_test,
         callbacks_train, callbacks_test,
         dataloader_train, dataloader_test,
         n_epoch, experiment_folder):

    json_filename = os.path.join(experiment_folder, 'output.json')
    json_logger = JSONLogger(json_filename)
    timer = Timer()

    for epoch in range(n_epoch):
        # start timer
        elapsed_time = {}
        timer.start()

        # train
        values_train = train_step(
            network,
            criterion_train,
            optimizer,
            dataloader_train,
            epoch,
            callbacks_train,
        )
        elapsed_time['train'] = timer.tick()

        # test
        values_test = test_step(
            network,
            criterion_test,
            dataloader_test,
            epoch,
            callbacks_test,
        )
        elapsed_time['test'] = timer.tick()

        # stats
        json_logger({'Epoch': epoch, 'Time (sec)': elapsed_time, **values_train, **values_test})
        plot.plot_results(json_filename)

        # save
        if epoch % 10 == 0:
            save_model(network, experiment_folder)

    save_model(network, experiment_folder)
