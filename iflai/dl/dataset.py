import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from skimage.util import crop, random_noise
import copy
import sys
from imblearn.over_sampling import RandomOverSampler
import os
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

sys.path.append("..")
seed_value = 42

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)

np.random.seed(seed_value)

torch.manual_seed(42)


def crop_pad_h_w(image_dummy, reshape_size):
    if image_dummy.shape[0] < reshape_size:
        h1_pad = (reshape_size - image_dummy.shape[0]) / 2
        h1_pad = int(h1_pad)
        h2_pad = reshape_size - h1_pad - image_dummy.shape[0]
        h1_crop = 0
        h2_crop = 0
    else:
        h1_pad = 0
        h2_pad = 0
        h1_crop = (reshape_size - image_dummy.shape[0]) / 2
        h1_crop = abs(int(h1_crop))
        h2_crop = image_dummy.shape[0] - reshape_size - h1_crop

    if image_dummy.shape[1] < reshape_size:
        w1_pad = (reshape_size - image_dummy.shape[1]) / 2
        w1_pad = int(w1_pad)
        w2_pad = reshape_size - w1_pad - image_dummy.shape[1]
        w1_crop = 0
        w2_crop = 0
    else:
        w1_pad = 0
        w2_pad = 0
        w1_crop = (reshape_size - image_dummy.shape[1]) / 2
        w1_crop = abs(int(w1_crop))
        w2_crop = image_dummy.shape[1] - reshape_size - w1_crop

    h = [h1_crop, h2_crop, h1_pad, h2_pad]
    w = [w1_crop, w2_crop, w1_pad, w2_pad]
    return h, w


def train_validation_test_split_wth_augmentation(X, y, validation_size=0.15, test_size=0.20, only_classes=None):
    train, test, y_train, _ = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    train, validation, _, _ = train_test_split(train, y_train, test_size=validation_size, stratify=y_train,
                                               random_state=42)
    return train, validation, test


class Dataset_Generator_Preprocessed_h5(Dataset):

    def __init__(self, path_to_data, set_indx, scaling_factor=4095., reshape_size=64, data_map=[],
                 transform=None, means=None, stds=None,
                 only_channels=[], channels_to_shuffle=[], only_classes=None, num_channels=12, return_only_image=False):

        self.path_to_data = path_to_data
        self.only_channels = only_channels
        self.channels_to_shuffle = channels_to_shuffle
        self.only_classes = only_classes
        self.object_numbers = set_indx

        self.scaling_factor = scaling_factor
        self.reshape_size = reshape_size
        self.data_map = data_map
        self.return_only_image = return_only_image

        self.num_channels = num_channels
        self.transform = transform
        if means is None:
            self.means = torch.zeros(self.num_channels)
        else:
            self.means = means
        if stds is None:
            self.stds = torch.ones(self.num_channels)
        else:
            self.stds = stds

    def __len__(self):
        return len(self.object_numbers)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        o_n = self.object_numbers[idx]
        try:
            r = h5py.File(os.path.join(self.path_to_data, '{}.h5'.format(o_n)), 'r')
            image_original = r.get('image')[()] / self.scaling_factor
            # convert str label to int
            label = r.get('label')[()]
            # creating the image
            h, w = crop_pad_h_w(image_original, self.reshape_size)
            h1_crop, h2_crop, h1_pad, h2_pad = h
            w1_crop, w2_crop, w1_pad, w2_pad = w
            image = np.zeros((self.num_channels, self.reshape_size, self.reshape_size), dtype=np.float64)
            nmb_of_channels = 0
            # filling the image with different channels
            for ch in range(image_original.shape[2]):
                if len(self.only_channels) == 0 or ch in self.only_channels:
                    image_dummy = crop(image_original[:, :, ch], ((h1_crop, h2_crop), (w1_crop, w2_crop)))
                    image_dummy = np.pad(image_dummy, ((h1_pad, h2_pad), (w1_pad, w2_pad)), "edge")
                    image[nmb_of_channels, :, :] = image_dummy
                    nmb_of_channels += 1
            image_original = None
            # map numpy array to tensor
            image = torch.from_numpy(copy.deepcopy(image))

            if self.transform:
                image = self.transform(image)

            for i in range(self.num_channels):
                image[i] = (image[i] - self.means[i]) / self.stds[i]

            if self.only_classes is not None:
                label = self.only_classes.index(label)
            label = np.array([self.data_map.get(label)])

            object_number = np.array([o_n])
            if len(self.channels_to_shuffle) > 0:
                for channel in self.channels_to_shuffle:
                    channel_shape = image[channel].shape
                    image[channel] = image[channel].flatten()[torch.randperm(len(image[channel].flatten()))].reshape(
                        channel_shape)

            sample = {'image': image, 'label': torch.from_numpy(label).long(), "idx": idx,
                      "object_number": object_number}

        except:
            sample = {'image': torch.from_numpy(
                np.zeros((self.num_channels, self.reshape_size, self.reshape_size), dtype=np.float64)),
                'label': torch.from_numpy(np.array([-1])).long(), "idx": idx, "object_number": np.array([-1])}
        if self.return_only_image:
            return sample["image"].float(), sample["label"][0]
        return sample
