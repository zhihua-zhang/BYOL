import os
from numpy import ceil, cos, pi
import torch
from utils import get_resnet
from dataset import prep_data, prep_data_cifar
from trainer import supervise_train, BYOL_pretrain
from models import SupervisedLightningModule


import argparse

BYOL_TRAIN_SIZE = 1e5 + 5e3


def get_args(args=None):
    parser = argparse.ArgumentParser(description='argparse')
    model_group = parser.add_argument_group(description='BYOL')
    model_group.add_argument("--run_mode", type=int, default=3,
                             help="which mode we are running:\
                                    no_pretrain:1,\
                                    BYOL_pretrain:2,\
                                    both_1and2: 3,\
                                    fine_tuning: 4")
    model_group.add_argument("--bz", type=int, default=1024,
                             help="mini-batch size")
    model_group.add_argument("--image_size", type=tuple, default=(96, 96),
                             help="input image size")
    model_group.add_argument("--byol_epochs", type=int, default=75,
                             help="number of BYOL epochs")
    model_group.add_argument("--base_lr", type=float, default=1e-4,
                             help="base learning rate")
    model_group.add_argument("--weight_decay", type=float, default=1e-6,
                             help="l2 weight decay")
    model_group.add_argument("--ft_ratio", type=float, default=1.0,
                             help="ratio of fine tuning dataset: [0.0, 1.0]")
    model_group.add_argument("--byol_ratio", type=float, default=1.0,
                             help="ratio of BYOL pre-training dataset: [0.0, 1.0]")
    model_group.add_argument("--lr_scheduler", type=list, default=[10, 20, 25],
                             help="lr scheduler")
    model_group.add_argument("--model_pth", type=str, default="",
                             help="reload model path")

    # Daniel's args
    model_group.add_argument("--target_decay_rate", type=float,
                             default=0.999, help="I think it's called `tau` in the paper")
    model_group.add_argument("--target_update_rule", type=int,
                             default=0, help="Index in list of update rules")
    model_group.add_argument("--option_3_cutoff", type=float,
                             default=0.5, help="Cutoff used for option 3 target_update_rule")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    args = get_args()

    print("="*28)
    print(f"run mode: {args.run_mode}")
    print(f"byol epoch: {args.byol_epochs}")
    print(f"byol batch size: {args.bz}")
    print(f"learning rate: {args.base_lr/128*args.bz}")
    print(f"weight decay: {args.weight_decay}")
    print(f"target (base) decay rate: {args.target_decay_rate}")
    print(f"target update rule: {args.target_update_rule}")
    print(f"byol ratio {args.byol_ratio}")
    print(f"fine tune ratio {args.ft_ratio}")
    print("="*28)

    train_loader, val_loader, byol_train_loader, byol_val_loader, finetune_train_loader = prep_data(
            args)

    if args.run_mode == 1:
        # Supervised training only
        model = get_resnet()
        supervised = SupervisedLightningModule(model, args)
        supervise_train(supervised, train_loader, val_loader, args)

    elif args.run_mode == 2:
        # BYOL pre-training
        if not os.path.exists("./pretrained.pth"):
            model = get_resnet()
            BYOL_pretrain(model, byol_train_loader, byol_val_loader, args)

    elif args.run_mode == 3:
        # Supervised training only
        model = get_resnet()
        supervised = SupervisedLightningModule(model, args)
        supervise_train(supervised, train_loader, val_loader, args)

        # (1) BYOL pre-training
        if not os.path.exists("./pretrained.pth"):
            model = get_resnet()
            BYOL_pretrain(model, byol_train_loader, byol_val_loader, args)

        # (2) Supervised fine-tuning
        model = get_resnet()
        model.load_state_dict(torch.load("./pretrained.pth"))
        supervised = SupervisedLightningModule(model, args)
        supervise_train(supervised, finetune_train_loader,
                        val_loader, args, with_pretrain=True)
    elif args.run_mode == 4:
        # Supervised fine-tuning with varying size
        if args.model_pth == "":
            print("pretrained model path not provided!")
        else:
            model = get_resnet()
            model.load_state_dict(torch.load(args.model_pth))
            supervised = SupervisedLightningModule(model, args)
            supervise_train(supervised, finetune_train_loader,
                            val_loader, args, with_pretrain=True)
    elif args.run_mode == 5:
        # Using STL10 to pretrain for CIFAR

        if args.model_pth == "":
            print("pretrained model path not provided!")
        # data prep
        train_loader, val_loader, byol_train_loader, byol_val_loader = prep_data_cifar(
            args)
        
        model = get_resnet()
        model.load_state_dict(torch.load(args.model_pth))
        supervised = SupervisedLightningModule(model, args)
        supervise_train(supervised, train_loader,
                        val_loader, args, with_pretrain=True)
    elif args.run_mode == 6:
        # Different target update rules
        tau_base = args.target_decay_rate
        K = ceil(args.byol_epochs * (BYOL_TRAIN_SIZE / args.bz))  # total epochs
        update_rules = [
            lambda _: tau_base,
            # the paper's update rule
            lambda k: 1 - (1 - tau_base) * \
            (cos(pi*k / K) + 1)/2,
            # inverse decay, go from base => 1/2
            lambda k: 1/2 + tau_base * (cos(pi*k / K) + 1)/4,
            # hard cutoff rule
            lambda k: tau_base if k < K * (args.option_3_cutoff) else 0
        ]
        # (1) BYOL pre-training
        if not os.path.exists("./pretrained.pth"):
            model = get_resnet()
            BYOL_pretrain(model, byol_train_loader, byol_val_loader, args,
                          target_update_rule=update_rules[args.target_update_rule])

        # (2) Supervised fine-tuning
        model = get_resnet()
        model.load_state_dict(torch.load("./pretrained.pth"))
        supervised = SupervisedLightningModule(model, args)
        supervise_train(supervised, train_loader,
                        val_loader, args, with_pretrain=True)
