import os
import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt


def read_data(path_to_data):
    X = []
    y = []
    for image_name in os.listdir(path_to_data):
        o_n = os.path.splitext(image_name)[0]
        r = h5py.File(os.path.join(path_to_data, image_name), 'r')
        X.append(int(o_n))
        y.append(r["label"][()])

    data_map = dict(zip(sorted(set(y)), np.arange(len(set(y)))))
    class_names = [i for i in list(data_map.keys())]
    return X, y, class_names, data_map


def get_statistics_h5(dataloader, only_channels, logging, num_channels):
    if len(only_channels) == 0:
        nmb_channels = num_channels
    else:
        nmb_channels = len(only_channels)

    statistics = dict()
    statistics["mean"] = torch.zeros(nmb_channels)
    statistics["std"] = torch.zeros(nmb_channels)

    for j, data_l in enumerate(dataloader, 0):
        data_l = data_l["image"].float()
        for n in range(nmb_channels):
            statistics["mean"][n] += data_l[:, n, :, :].mean()
            statistics["std"][n] += data_l[:, n, :, :].std()
    statistics["mean"] = statistics["mean"].div_(len(dataloader))
    statistics["std"] = statistics["std"].div_(len(dataloader))
    if logging is not None:
        logging.info('statistics used: %s' % (str(statistics)))
    return statistics


def calculate_weights(y_train):
    class_sample_count = np.array([len(np.where(np.asarray(y_train) == t)[0]) for t in np.unique(y_train)])
    weights = len(y_train) / class_sample_count
    return weights


def plot_heatmap_3_channels(heatmap):
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 15))
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.imshow(heatmap[0])
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.imshow(heatmap[1])
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)
    ax3.imshow(heatmap[2])
