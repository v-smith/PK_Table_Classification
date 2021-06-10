'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet
complete list of built-in transforms:
https://pytorch.org/docs/stable/torchvision/transforms.html
'''

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

class WineDataset(Dataset):

    def __init__(self, transform=None):
        xy = np.loadtxt('../data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # note that we do not convert to tensor here
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, [0]]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

dataset= WineDataset(transform=None)
first_data= dataset[0]
features, labels= first_data
print(features)
print(type(features), type(labels))

composed= torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset= WineDataset(transform= composed)
first_data= dataset[0]
features, labels= first_data
print(features)
print(type(features), type(labels))
