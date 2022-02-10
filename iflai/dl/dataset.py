import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from skimage.util import crop
import copy
import os
import numpy as np

seed_value = 42

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)


def get_image(h5_file_path):
    h5_file = h5py.File(h5_file_path, 'r')
    image_original =h5_file.get('image')[()]
    h5_file.close()
    h5_file = None
    return image_original

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


class DatasetGenerator(Dataset):

    def __init__(self,
                metadata,
                reshape_size=64,
                label_map=[],
                task="classification",
                transform=None,
                selected_channels=["Ch1"]):

        self.metadata = metadata.copy().reset_index(drop = True)
        self.selected_channels = selected_channels
        self.num_channels = len(self.selected_channels)
        self.task = task
        self.reshape_size = reshape_size
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## get image and label
        h5_file_path = self.metadata.loc[idx,"file"]
        image_original= get_image(h5_file_path)
        label = self.metadata.loc[idx,"label"]

        ## creating the image
        h, w = crop_pad_h_w(image_original, self.reshape_size)
        h1_crop, h2_crop, h1_pad, h2_pad = h
        w1_crop, w2_crop, w1_pad, w2_pad = w
        image = np.zeros((  self.num_channels,
                            self.reshape_size,
                            self.reshape_size),
                            dtype=np.float64)

        # filling the image with selected channels
        for i, ch in enumerate(self.selected_channels):
            image_dummy = crop( image_original[:, :, ch],
                                ((h1_crop, h2_crop),
                                (w1_crop, w2_crop)))
            image_dummy = np.pad(image_dummy,
                                ((h1_pad, h2_pad),(w1_pad, w2_pad)),
                                "constant",
                                constant_values = image_dummy.mean())
            image[i, :, :] = image_dummy
            image_dummy = None

        image_original = None

        # map numpy array to tensor
        image = torch.from_numpy(copy.deepcopy(image))

        if self.transform:
            image = self.transform(image)

        if self.task == "classification":
            # preparing the label part
            label = self.label_map[label]
            label = torch.tensor(label).long()
            return image.float(),  label
        elif self.task == "autoencoder":
            # preparing the label part
            label = image
            return image.float(),  image.float()
        else:
            raise KeyError("%s is not a correct task" % self.task)


class DatasetGeneratorFixMatch(Dataset):

    def __init__(self,
                metadata,
                label_map,
                reshape_size=64,
                weak_transform=None,
                strong_transform=None,
                selected_channels=["Ch1"]):

        self.metadata = metadata.copy().reset_index(drop = True)
        self.selected_channels = selected_channels
        self.num_channels = len(self.selected_channels)
        self.reshape_size = reshape_size
        self.label_map = label_map
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## get image and label
        h5_file_path = self.metadata.loc[idx,"file"]
        image_original= get_image(h5_file_path)
        label = self.metadata.loc[idx,"label"]

        ## creating the image
        h, w = crop_pad_h_w(image_original, self.reshape_size)
        h1_crop, h2_crop, h1_pad, h2_pad = h
        w1_crop, w2_crop, w1_pad, w2_pad = w
        image = np.zeros((  self.num_channels,
                            self.reshape_size,
                            self.reshape_size),
                            dtype=np.float64)

        # filling the image with selected channels
        for i, ch in enumerate(self.selected_channels):
            image_dummy = crop( image_original[:, :, ch],
                                ((h1_crop, h2_crop),
                                (w1_crop, w2_crop)))
            image_dummy = np.pad(image_dummy,
                                ((h1_pad, h2_pad),(w1_pad, w2_pad)),
                                "constant",
                                constant_values = image_dummy.mean())
            image[i, :, :] = image_dummy
            image_dummy = None

        image_original = None

        # map numpy array to tensor
        image = torch.from_numpy(copy.deepcopy(image))
        image_w = image.detach().clone()
        image_s = image.detach().clone()
        image = None

        if self.weak_transform:
            image_w = self.weak_transform(image_w)

        if self.strong_transform:
            image_s = self.strong_transform(image_s)


        label = self.label_map[label]
        label = torch.tensor(label).long()
        return [image_w.float(), image_s.float()],  label


class DatasetGeneratorPreprocessedH5(Dataset):

    def __init__(self, path_to_data, set_indx, scaling_factor=4095., reshape_size=64, data_map=[],
                 transform=None, means=None, stds=None,
                 only_channels=[], channels_to_shuffle=[], perturb=False, only_classes=None, num_channels=12,
                 return_only_image=False):

        self.path_to_data = path_to_data
        self.only_channels = only_channels
        self.channels_to_shuffle = channels_to_shuffle
        self.only_classes = only_classes
        self.object_numbers = set_indx

        self.scaling_factor = scaling_factor
        self.reshape_size = reshape_size
        self.data_map = data_map
        self.return_only_image = return_only_image
        self.perturb = perturb
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
            mask_formatted = False
            if 'mask' not in r.keys() or isinstance(r.get('mask')[()], h5py._hl.base.Empty):
                mask_original = torch.from_numpy(
                    np.zeros((self.num_channels, self.reshape_size, self.reshape_size), dtype=np.float64))
                mask_formatted = True
            else:
                mask_original = r.get('mask')[()]
            # else:
            #    mask_original = mask_original.T
            # creating the image
            h, w = crop_pad_h_w(image_original, self.reshape_size)
            h1_crop, h2_crop, h1_pad, h2_pad = h
            w1_crop, w2_crop, w1_pad, w2_pad = w
            image = np.zeros((self.num_channels, self.reshape_size, self.reshape_size), dtype=np.float64)
            mask = np.zeros((self.num_channels, self.reshape_size, self.reshape_size), dtype=np.float64)
            nmb_of_channels = 0
            # filling the image with different channels
            for ch in range(image_original.shape[2]):
                if len(self.only_channels) == 0 or ch in self.only_channels:
                    image_dummy = crop(image_original[:, :, ch], ((h1_crop, h2_crop), (w1_crop, w2_crop)))
                    image_dummy = np.pad(image_dummy, ((h1_pad, h2_pad), (w1_pad, w2_pad)), "edge")
                    image[nmb_of_channels, :, :] = image_dummy

                    # do the same for the mask
                    if not mask_formatted:
                        mask_dummy = crop(mask_original[:, :, ch], ((h1_crop, h2_crop), (w1_crop, w2_crop)))
                        mask_dummy = np.pad(mask_dummy, ((h1_pad, h2_pad), (w1_pad, w2_pad)), "edge")
                        mask[nmb_of_channels, :, :] = mask_dummy
                        nmb_of_channels += 1
                    else:
                        mask[nmb_of_channels, :, :] = mask_original[nmb_of_channels, :, :]
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
                    if self.perturb:
                        image[channel] = torch.full(channel_shape, torch.mean(image[channel]))
                    else:
                        image[channel] = image[channel].flatten()[
                            torch.randperm(len(image[channel].flatten()))].reshape(
                            channel_shape)

            sample = {'image': image, 'mask': mask, 'label': torch.from_numpy(label).long(), "idx": idx,
                      "object_number": object_number}
            r.close()

        except:
            sample = {'image': torch.from_numpy(
                np.zeros((self.num_channels, self.reshape_size, self.reshape_size), dtype=np.float64)),
                'mask': torch.from_numpy(
                    np.zeros((self.num_channels, self.reshape_size, self.reshape_size), dtype=np.float64)),
                'label': torch.from_numpy(np.array([-1])).long(), "idx": idx, "object_number": np.array([-1])}
        if self.return_only_image:
            return sample["image"].float(), sample["label"][0]
        return sample
