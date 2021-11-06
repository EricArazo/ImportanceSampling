import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from os.path import join
import csv
from tqdm import tqdm

def get_dataset(args, transform_train, transform_test):
    train_data, train_labels, val_data, val_labels = make_dataset(args)
    trainset = MiniImagenet84(args, train_data, train_labels, train = "True", transform=transform_train)
    testset = MiniImagenet84(args, val_data, val_labels, train = "False", transform=transform_test)
    return trainset, testset


class MiniImagenet84(Dataset):
    def __init__(self, args, data, labels, train = None, transform=None, target_transform=None, sample_indexes = None, download=False):
        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.data = data
        self.labels = labels
        self.train_data = self.data
        self.train_labels = self.labels

        self.args = args
        if sample_indexes is not None:
            self.train_data = self.train_data[sample_indexes]
            self.train_labels = np.array(self.train_labels)[sample_indexes]
        self.num_classes = self.args.num_classes
        self.train_samples_idx = []
        self.train_probs = np.ones(len(self.labels))*(-1)
        self.avg_probs = np.ones(len(self.labels))*(-1)
        self.times_seen = np.ones(len(self.labels))*1e-6


    def __getitem__(self, index):
        img, labels = self.data[index], self.labels[index]

        if self.args.pre_load == "True":
            img = Image.fromarray(img, mode='RGB')
        else:
            img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        if self.train == "True":
            return img, labels, index
        elif self.train == "False":
            return img, labels

    def __len__(self):
        return len(self.data)


def make_dataset(args):
    np.random.seed(args.seed_dataset)
    csv_files = ["train.csv", "val.csv", "test.csv"]
    img_paths = []
    labels = []
    for split in csv_files:
        in_csv_path = join(args.train_root, split)
        in_images_path = join(args.train_root, "images")

        with open(in_csv_path) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)
            for i,row in enumerate(csvreader):
                img_paths.append(join(in_images_path,row[0]))
                labels.append(row[1])

    mapping = {y: x for x, y in enumerate(np.unique(labels))}
    label_mapped = [mapping[i] for i in labels]

    # split in train and validation:
    train_num = 50000
    val_num = 10000

    idxes = np.random.permutation(len(img_paths))

    img_paths = np.asarray(img_paths)[idxes]
    label_mapped = np.asarray(label_mapped)[idxes]

    train_img_paths = img_paths[:train_num]
    train_labels = label_mapped[:train_num]
    val_img_paths = img_paths[train_num:]
    val_labels = label_mapped[train_num:]

    if args.pre_load == "True":
        train_pil_images = []
        print("Loading Images in memory...")
        for i in tqdm(train_img_paths):
            train_pil_images.append(np.asarray(Image.open(i)))
        train_data = np.asarray(train_pil_images)

        val_pil_images = []
        for i in val_img_paths:
            val_pil_images.append(np.asarray(Image.open(i)))
        val_data = np.asarray(val_pil_images)
    else:
        train_data = train_img_paths
        val_data = val_img_paths

    return train_data, train_labels, val_data, val_labels
