import torch
import numpy as np
from torch import optim
import argparse
import logging
import os
import time
import random
import sys
import json

from dataset.load_dataset import data_config

sys.path.append('../utils_train')
from utils_train import *
from resnet import ResNet18

logger = logging.getLogger()
logger.setLevel(logging.INFO)

seeds_pool = [66446573, 59719201, 37273794, 40015061, 72942995, 10000149, 93149410, 10809540, 62353799, 84783536, 51422067]

def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    
    parser.add_argument('--batch_size', type=int, default=128, help='#images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    parser.add_argument('--num_workers', type=int, default=8, help="How many samples to remove")

    parser.add_argument('--dataset', type=str, default='cifar10', help='Choose dataset: mini-imagenet, svhn, cifar10, or cifar100')
    parser.add_argument('--pre_load', type=str, default='True', help='Load full mini-Imagenet in RAM')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of in-distribution classes')
    parser.add_argument('--train_root', default='./data', help='root for train data')
    parser.add_argument('--download', type=bool, default=True, help='download dataset')
    parser.add_argument('--validation_exp', type=str, default='False', help='Hold a subset of samples from the training set for validation')
    parser.add_argument('--val_samples', type=int, default=5000, help='Number of samples to be kept for validation')

    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--gamma_scheduler', default=0.1, type=float, help='Value to decay the learning rate')
    parser.add_argument('--scheduler_type', type=str, default="linear", help='Choose type of scheduler: step, or linear')
    parser.add_argument('--epoch', type=int, default=200, help='training epoches')
    parser.add_argument('--M', action='append', type=int, default=[], help="Milestones for the LR sheduler")
    parser.add_argument('--experiment_name', type=str, default = 'Proof',help='name of the experiment (for the output files)')
    
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--seed_initialization', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--seed_dataset', type=int, default=42, help='random seed (default: 42)')
    
    parser.add_argument('--budget', type=float, default=1.0, help='Percentage of buget to use')
    parser.add_argument('--method', type=str, default='None', help='Training strategy: SGD, unif-SGD, p-SGD, c-SGD, or selective_backpropagation')
    parser.add_argument('--augmentation', type=str, default='CE', help='Choose the augmentation: standard, mixup, ricap, or randaugment')
    parser.add_argument('--alpha', type=float, default=0.3, help='Value for mixup and RICAP')
    parser.add_argument('--randaugm_N', type=int, default=2, help="Number of augmentations applied at once")
    parser.add_argument('--randaugm_M', type=int, default=4, help="Strength of the augmentations applied")
    parser.add_argument('--c_sgd_warmup', type=int, default=0, help="Number of ecpochs with random sampling for p-SGD andd c-SGD")
    
    # Arguments for the selective backprop paper
    parser.add_argument('--SB_sensitivity', type=float, default=3.0, help='sensitivity of SB')
    parser.add_argument('--SB_strategy', type=str, default="sb", help='Select the strategy from the SB paper')
    parser.add_argument('--SB_init', type=str, default="False", help='Do SB the first epochs')

    args = parser.parse_args()

    return args



def main(args):
    # Initializing seeds
    args.seed_initialization = seeds_pool[args.seed_initialization]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_dev)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed_initialization)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed_initialization)  # GPU seed
    random.seed(args.seed_initialization)  # python seed for image transformation
    np.random.seed(args.seed_initialization)

    # Create data loaders
    train_loader, test_loader = data_config(args)
    
    # Create model
    model = ResNet18(num_classes=args.num_classes).to(device)
    print('Total params: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.M, gamma=args.gamma_scheduler)

    loss_train_epoch = []
    loss_val_epoch = []
    acc_train_per_epoch = []
    acc_val_per_epoch = []

    exp_path = os.path.join('./', 'models' + '_{0}'.format(args.experiment_name))
    res_path = os.path.join('./', 'metrics' + '_{0}'.format(args.experiment_name))

    if not os.path.isdir(res_path):
        os.makedirs(res_path)

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)

    cont = 0
    initial_epoch = 1
    total_epochs = args.epoch

    ############################## Initialize the Selective Backpropagation method ###################################
    # code from here: https://github.com/angelajiang/SelectiveBackprop
    if args.method == "selective_backpropagation":
        sys.path.append('./SelectiveBackprop')
        from lib.SelectiveBackpropper import SelectiveBackpropper

        ## setup learning rate
        num_samples = len(train_loader.dataset)
        it_per_epoch = len(train_loader)

        if args.scheduler_type == "linear":
            lr_path = './lr_file_lin.json'
            lr_values = np.linspace(args.lr, 1e-6, args.epoch*it_per_epoch)
            lr_steps = np.asarray(range(0,args.epoch*it_per_epoch))*args.batch_size
            lr_dict = {}
            for i, stp in enumerate(lr_steps):
                lr_dict[str(stp)] = lr_values[i]
        elif args.scheduler_type == "standard":
            lr_path = './lr_file_step.json'
            lr_dict = {}
            lr_dict[str(0)] = args.lr
            lr_dict[str(args.M[0]*it_per_epoch*args.batch_size)] = args.lr/10
            lr_dict[str(args.M[1]*it_per_epoch*args.batch_size)] = args.lr/100

        with open(lr_path, 'w') as fp:
            json.dump(lr_dict, fp)

        if args.augmentation == "RICAP":
            train_loader.dataset.ricap = "True"
            train_loader.dataset.mixup = "False"
            ricap = "True"
            mixup = "False"
        elif args.augmentation == "mixup":
            train_loader.dataset.mixup = "True"
            train_loader.dataset.ricap = "False"
            ricap = "False"
            mixup = "True"
        else:
            train_loader.dataset.ricap = "False"
            train_loader.dataset.mixup = "False"
            ricap = "False"
            mixup = "False"

        sb = SelectiveBackpropper(model, optimizer, args.SB_sensitivity,
                          args.batch_size, lr_path, args.num_classes,
                          num_samples, True, args.SB_strategy,
                          'relative', 'alwayson', 1,
                          ricap, mixup)
    ###########################################################################################


    st = time.time()

    for epoch in range(initial_epoch, total_epochs + 1):
        # Selecting samples
        print('######## Selecting samples for training ########')
        if not args.method == "SGD" and not args.method == "selective_backpropagation":
            train_loader = prepare_loader(args, train_loader, epoch)

        # Train
        print('######## Training ########')
        print("=================> Name: {} --- Budget: {}".format(args.experiment_name, args.budget))
        if args.method == "selective_backpropagation":
            sb.trainer.train(train_loader)
            top1_train_ac = sb.logger.partition_accuracy
            loss_per_epoch = sb.logger.partition_loss
            sb.next_epoch()
            sb.next_partition()
        else:
            loss_per_epoch, _, top1_train_ac = train_CrossEntropy(args, model, device, \
                                                        train_loader, optimizer, epoch)
        loss_train_epoch += [loss_per_epoch]

        # Test
        print('######## Test ########')
        if args.method == "selective_backpropagation":
            acc_val_per_epoch_i, loss_per_epoch = test_sb(test_loader, epoch, sb, model)
        else:
            loss_per_epoch, acc_val_per_epoch_i = testing(args, model, device, test_loader)

        loss_val_epoch += loss_per_epoch
        acc_train_per_epoch += [top1_train_ac]
        acc_val_per_epoch += acc_val_per_epoch_i
        print('Epoch time: {:.2f} seconds\n'.format(time.time()-st))
        st = time.time()

        # Saving
        print('######## Saving model and metrics ########')
        if epoch == initial_epoch:
            best_acc_val = acc_val_per_epoch_i[-1]
            snapBest = 'best_epoch_%d_valAcc_%.2f_budget_%.2f_bestAccVal_%.2f' % (
                epoch, acc_val_per_epoch_i[-1], args.budget, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
        else:
            if acc_val_per_epoch_i[-1] > best_acc_val:
                best_acc_val = acc_val_per_epoch_i[-1]
                if cont > 0:
                    try:
                        os.remove(os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
                        os.remove(os.path.join(exp_path, snapBest + '.pth'))
                    except OSError:
                        pass
                snapBest = 'best_epoch_%d_valAcc_%.2f_budget_%.2f_bestAccVal_%.2f' % (
                    epoch, acc_val_per_epoch_i[-1], args.budget, best_acc_val)
                torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
                torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))

        cont += 1

        if epoch == total_epochs:
            snapLast = 'last_epoch_%d_valAcc_%.2f_budget_%d_bestValLoss_%.2f' % (
                epoch, acc_val_per_epoch_i[-1], args.budget, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapLast + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapLast + '.pth'))

        # Save losses:
        np.save(res_path + '/' + 'LOSS_epoch_train.npy', np.asarray(loss_train_epoch))
        np.save(res_path + '/' + 'LOSS_epoch_val.npy', np.asarray(loss_val_epoch))

        # Save accuracies:
        np.save(res_path + '/' + 'accuracy_per_epoch_train.npy', np.asarray(acc_train_per_epoch))
        np.save(res_path + '/' + 'accuracy_per_epoch_val.npy', np.asarray(acc_val_per_epoch))

        if not args.scheduler_type == "linear":
            scheduler.step()

    print('Best ac:%f' % best_acc_val)


if __name__ == "__main__":
    args = parse_args()
    logging.info(args)

    main(args)
