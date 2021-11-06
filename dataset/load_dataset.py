import torch 
import torchvision.transforms as transforms

from dataset.cifar10_dataset import get_dataset as get_dataset_cifar10
from dataset.cifar100_dataset import get_dataset as  get_dataset_cifar100
from dataset.svhn_dataset import get_dataset as  get_dataset_svhn
from dataset.miniImagenet_dataset import get_dataset as  get_dataset_miniImagenet

import sys
sys.path.append('./utils_train')
from randAugm import RandAugment

def data_config(args):


    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        size = 32
        flip = 0.5
    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        size = 32
        flip = 0.5
    elif args.dataset == 'svhn':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
        size = 32
        flip = 0.0
    elif args.dataset == 'mini-imagenet':
        mean = [0.4728, 0.4487, 0.4031]
        std = [0.2744, 0.2663 , 0.2806]
        size = 84
        flip = 0.5

    transform_train = transforms.Compose([
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(flip),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if args.augmentation == "randaugment":
        transform_train.transforms.insert(0, RandAugment(args.randaugm_N, args.randaugm_M))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if args.dataset == 'cifar10':
        trainset, testset = get_dataset_cifar10(args, transform_train, transform_test)
    elif args.dataset == 'cifar100':
        trainset, testset = get_dataset_cifar100(args, transform_train, transform_test)
    elif args.dataset == 'svhn':
        trainset, testset = get_dataset_svhn(args, transform_train, transform_test)
    elif args.dataset == 'mini-imagenet':
        trainset, testset = get_dataset_miniImagenet(args, transform_train, transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, \
                                                shuffle=True, \
                                                num_workers=args.num_workers, \
                                                pin_memory=True, \
                                                drop_last = True)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print('############# Data loaded #############')
    return train_loader, test_loader

