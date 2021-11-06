import os
import torchvision as tv
import numpy as np
from PIL import Image


def get_dataset(args, transform_train, transform_test):
    cifar_train = Cifar100Train(args, train=True, transform=transform_train, download = args.download)
    testset = tv.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    return cifar_train, testset

class Cifar100Train(tv.datasets.CIFAR100):
    def __init__(self, args, train=True, transform=None, target_transform=None, download=False):
        super(Cifar100Train, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.target_transform = target_transform

        self.args = args
        self.num_classes = self.args.num_classes

        self.data = self.train_data
        self.labels = np.asarray(self.train_labels, dtype=np.long)

        self.train_samples_idx = []
        self.train_probs = np.ones(len(self.labels))*(-1)
        self.avg_probs = np.ones(len(self.labels))*(-1)
        self.times_seen = np.ones(len(self.labels))*1e-6

    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return img, labels, index
