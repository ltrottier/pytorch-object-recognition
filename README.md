# pytorch-object-recognition
Minimalist PyTorch convolutional neural network implementation for object recognition.

## Requirements

#### Create virtual environment

```bash
conda create --name pytorch python=3
source activate pytorch
```

#### Install packages
1) [pytorch](http://pytorch.org/) (tested with version 0.2.0.post3)
2) `pip install -r requirements.txt` 

## Instructions

The overall procedure to train a network is divided into two parts:
1) Create an opts file.
2) Train with that opts file.

#### Create opts file

Use `opts.py` to generate an opts file. By default, this will create an opts file for training a simple residual network on the CIFAR10 dataset (which will be automatically downloaded).

The script can parse arguments so you can generate multiple opts file as you want (see script for the list of available arguments):
```bash
python opts.py --option-type option-value
```

The opts file is saved as `opts.txt` in folder designated by `experiment_folder`, which will be created if it does not exist. By default, `experiment_folder=results/exp1`.

#### Train the network

Use `main.py` with the previously generated opts file. For instance:
```bash
python main.py results/exp1/opts.txt
```
will start training a residual network in CIFAR10.

