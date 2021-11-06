# Example of unif-SGD on CIFAR-10 with 30% of the budget and RICAP data augmentation
python3 train.py --epoch 200 \
--dataset "cifar10" --num_classes 10 \
--augmentation "ricap" --method "unif-SGD" --budget 0.3 \
--experiment_name unif-SGD_cifar10_ricap_b03


# Example of p-SGD on CIFAR-100 with 30% of the budget and MixUp data augmentation
python3 train.py --epoch 200 \
--dataset "cifar100" --num_classes 100 \
--augmentation "mixup" --method "p-SGD" --budget 0.3 \
--experiment_name p-SGD_cifar100_mixup_b03


# Example of p-SGD on SVHN with 30% of the budget and RandAugment data augmentation
python3 train.py --epoch 200 \
--dataset "svhn" --num_classes 10 \
--augmentation "randaugment" --method "p-SGD" --budget 0.3 \
--experiment_name p-SGD_svhn_randaugment_b03


# Example of unif-SGD on mini-ImageNet with 30% of the budget and standard data augmentation
python3 train.py --epoch 200 \
--dataset "mini-imagenet" --num_classes 100 \
--augmentation "standard" --method "unif-SGD" --budget 0.3 \
--experiment_name unif-SGD_mini-imagenet_standard_b03