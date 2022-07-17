from __future__ import print_function

import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    data_folder = './data'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

class STL10Instance(datasets.STL10):
    """CIFAR100Instance Dataset.
    """
    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


def get_stl10_dataloaders(batch_size=128, num_workers=8, is_instance=False, is_retrieval=False, augmentation='auto'):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = STL10Instance(root=data_folder,
                                      download=True,
                                      split='train',
                                      transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    n_data = len(train_set)
    test_set = datasets.STL10(root=data_folder,
                                 download=True,
                                 split='test',
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader

