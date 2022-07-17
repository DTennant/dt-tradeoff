"""
get data loaders
"""
from __future__ import print_function

import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from .autoaugment import CIFAR10Policy


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    data_folder = './data/CINIC-10'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class ImageFolderSample(datasets.ImageFolder):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """
    def __init__(self, root, transform=None, target_transform=None,
                 is_sample=False, k=4096):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        self.k = k
        self.is_sample = is_sample

        print('stage1 finished!')

        if self.is_sample:
            num_classes = len(self.classes)
            num_samples = len(self.samples)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                path, target = self.imgs[i]
                label[i] = target

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]

        print('dataset initialized!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


def get_cinic10_dataloader_sample(dataset='cinic10', batch_size=128, num_workers=8, is_sample=True, k=4096):
    """Data Loader for ImageNet"""

    if dataset == 'cinic10':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    # add data transform
    normalize = transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                     std=[0.24205776, 0.23828046, 0.25874835])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'valid')

    train_set = ImageFolderSample(train_folder, transform=train_transform, is_sample=is_sample, k=k)
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    print('num_samples', len(train_set.samples))
    print('num_class', len(train_set.classes))

    return train_loader, test_loader, len(train_set)


def get_cinic10_dataloader(dataset='cinic10', batch_size=128, num_workers=16, is_instance=False, augmentation='none'):
    """
    Data Loader for imagenet
    """
    if dataset == 'cinic10':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    normalize = transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                     std=[0.24205776, 0.23828046, 0.25874835])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])
    if augmentation == 'transfer':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])
    if augmentation == 'mine':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    if augmentation == 'auto':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'valid')

    if is_instance:
        train_set = ImageFolderInstance(train_folder, transform=train_transform, target_transform=train_transform)
    else:
        train_set = datasets.ImageFolder(train_folder, transform=train_transform, target_transform=None)

    test_set = datasets.ImageFolder(test_folder, transform=test_transform)
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers//2,
                             pin_memory=True)

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader
