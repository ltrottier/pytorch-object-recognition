import os
import numpy as np
import pandas as pd
from skimage import io, transform

from PIL import Image
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision

from copy import deepcopy


# processing

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img = sample['img']
        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        # image
        img_dtype = img.dtype
        sample['img'] = transform.resize(
            img, (new_h, new_w), mode='constant').astype(img_dtype)

        return sample


class RandomTranslation(object):
    """Translate the image and its attributes in a sample. Padding is 0.

    Args:
        max_offset: nb of pixel for translation.
    """

    def __init__(self, max_offset):
        self.max_offset = max_offset

    def translate(self, img, tx, ty):
        h, w = img.shape[:2]
        # in timg
        timg_x1 = np.maximum(tx, 0)
        timg_x2 = np.minimum(w + tx, w) - 1
        timg_y1 = np.maximum(ty, 0)
        timg_y2 = np.minimum(h + ty, h) - 1
        # in img
        img_x1 = np.maximum(-tx, 0)
        img_x2 = np.minimum(w - tx, w) - 1
        img_y1 = np.maximum(-ty, 0)
        img_y2 = np.minimum(h - ty, h) - 1
        # translation
        timg = np.zeros(img.shape).astype(img.dtype)
        timg[timg_y1:timg_y2, timg_x1:timg_x2, :] = img[img_y1:img_y2,
                                                        img_x1:img_x2, :]
        return timg

    def __call__(self, sample):
        img = sample['img']

        tx = np.random.randint(-self.max_offset, self.max_offset)
        ty = np.random.randint(-self.max_offset, self.max_offset)

        h, w = img.shape[:2]

        # image
        sample['img'] = self.translate(img, tx, ty)

        return sample


class SquareCrop(object):
    """Crop the image in a sample. Can be random.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, random):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.random = random

    def __call__(self, sample):
        img = sample['img']

        h, w = img.shape[:2]
        new_h, new_w = self.output_size
        if self.random:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
        else:
            top = (h - new_h) // 2
            left = (w - new_w) // 2

        # image
        sample['img'] = img[top: top + new_h, left: left + new_w]

        return sample


class RandomHFlip(object):
    """ Randomly flip the image and its attributes horizontaly.
    """

    def __call__(self, sample):
        img = sample['img']
        h, w = img.shape[:2]

        if np.random.rand() > 0.5:
            # image
            sample['img'] = img[:, ::-1, :]

        return sample


class ColorJitter(object):
    """ Change contrast, saturation and brightness
    """

    def __init__(self, contrast_var, saturation_var, brightness_var):
        self.contrast_var = contrast_var
        self.saturation_var = saturation_var
        self.brightness_var = brightness_var

    def blend(self, img1, img2, alpha):
        return img1 * alpha + img2 * (1 - alpha)

    def grayscale(self, img):
        gray = img[:, :, 0:1] * 0.299 + img[:, :, 1:2] * \
            0.587 + img[:, :, 2:3] * 0.114
        gray = np.concatenate([gray, gray, gray], 2)
        return gray

    def saturation(self, img):
        gray = self.grayscale(img)
        alpha = 1 + \
            np.random.uniform(-self.saturation_var, self.saturation_var)
        sat = self.blend(img, gray, alpha)
        return sat

    def brightness(self, img):
        alpha = 1 + \
            np.random.uniform(-self.brightness_var, self.brightness_var)
        bright = self.blend(img, np.zeros(img.shape), alpha)
        return bright

    def contrast(self, img):
        gray = self.grayscale(img)
        gray.fill(gray.mean())
        alpha = 1 + np.random.uniform(-self.contrast_var, self.contrast_var)
        cont = self.blend(img, gray, alpha)
        return cont

    def __call__(self, sample):
        img = sample['img']

        trs = [self.saturation, self.brightness, self.contrast]
        np.random.shuffle(trs)
        for tr in trs:
            img = tr(img)
        img = img.astype(np.float32)
        sample['img'] = img

        return sample


class Normalize(object):
    """ Normalize image with mean and standard deviation (per channel).
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, sample):
        sample['img'] = (sample['img'] - self.mean) / self.std
        return sample


class TransposeImage(object):
    """Convert numpy image to torch image."""

    def __call__(self, sample):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        sample['img'] = sample['img'].transpose((2, 0, 1))
        return sample


class ToTensor(object):
    """Convert numpy arrays to torch tensors."""

    def __call__(self, sample):
        for key in sample.keys():
            sample[key] = torch.from_numpy(sample[key])

        return sample


class DeepCopy(object):
    """Deep copy all numpy arrays."""

    def __call__(self, sample):
        return deepcopy(sample)


class SampleAsInputTargetMeta(object):
    """ Convert the sample dictionary into a pair (input, target, meta)
    """

    def __call__(self, sample):
        input = sample.pop('img', None)
        target = sample.pop('label', None)
        return (input, target, sample)


# datasets

# dummy
def load_dummy_dataset(
        dataset_dir,
        train_test_ratio,
        batch_size,
        shuffle,
        num_workers,
        drop_last):

    def load_data_list(dataset_dir):
        filepath = os.path.join(dataset_dir, 'data_list.npz')
        data_list = np.load(filepath)['data_list']

        # replace filepath
        for entry in data_list:
            entry['filepath'] = os.path.join(dataset_dir, entry['filepath'])

        return data_list

    class DummyDataset(Dataset):
        def __init__(self, data_list, processing):
            self.data_list = data_list
            self.processing = processing

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            entry = self.data_list[idx]

            # image
            img = np.array(Image.open(entry['filepath']).convert('RGB'))
            img = img.astype(np.float32) / 255

            # label
            label = np.array([entry['label']]).astype(int)

            # idx
            index = np.array([idx])

            sample = {
                'index': index,
                'img': img,
                'label': label
            }

            if self.processing is not None:
                sample = self.processing(sample)

            return sample

    def create_dataset_and_dataloader(data_list, processing):
        dataset = DummyDataset(data_list, processing)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=drop_last)

        return dataset, dataloader

    # raw
    data_list = load_data_list(dataset_dir)
    cutoff = int(len(data_list) * train_test_ratio)

    # train
    processing_train = torchvision.transforms.Compose([
        DeepCopy(),
        Rescale((32, 32)),
        RandomTranslation(4),
        ColorJitter(0.4, 0.4, 0.4),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        TransposeImage(),
        ToTensor(),
        SampleAsInputTargetMeta(),
    ])
    data_list_train = data_list[:cutoff]
    dataset_train, dataloader_train = create_dataset_and_dataloader(
        data_list_train, processing_train)

    # test
    processing_test = torchvision.transforms.Compose([
        DeepCopy(),
        Rescale((32, 32)),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        TransposeImage(),
        ToTensor(),
        SampleAsInputTargetMeta(),
    ])
    data_list_test = data_list[cutoff:]
    dataset_test, dataloader_test = create_dataset_and_dataloader(
        data_list_test, processing_test)

    return (dataset_train, dataset_test), (dataloader_train, dataloader_test)


# Utils

def initialize(
        dataset_name,
        dataset_dir,
        train_test_ratio,
        batch_size,
        shuffle,
        num_workers,
        drop_last):

    if dataset_name == 'dummy':
        dataset_train_test, dataloader_train_test = load_dummy_dataset(
            dataset_dir, train_test_ratio, batch_size, shuffle, num_workers, drop_last)
    else:
        raise Exception("Invalid dataset type: {}".format(dataset_type))

    return dataset_train_test, dataloader_train_test


def compute_channel_mean_std(dataloader):
    sm = np.zeros(3)
    sm2 = np.zeros(3)
    n = np.zeros(3)

    for i, batch in enumerate(dataloader):
        img = batch['img'].numpy()
        sm = sm + np.sum(img, axis=(0, 1, 2))
        sm2 = sm2 + np.sum(img**2, axis=(0, 1, 2))
        n = n + np.prod(img.shape[:3])

    mean = sm / n
    std = np.sqrt(sm2 / n - mean**2)
    return mean, std
