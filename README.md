# Understanding Self-Supervised Learning with Bootstrap-Your-Own-Latent (BYOL)

Zhihua Zhang, Ramya Muthukrishnan, Daniel Lee

## Requirements

Running the code requires Pytorch, Torchvision, Kornia, Pytorch Lightning libraries in Python 3

## Running Supervised Learning

To run supervised training without BYOL, run:

`python3 main.py --run_mode 1`

## Running Self-Supervised Learning

To run only self-supervised pre-training, run:

`python3 main.py --run_mode 2`

To run supervised training without BYOL, self-supervised pre-training, and then supervised fine-tuning, run:

`python3 main.py --run_mode 3`

To run supervised fine-tuning with varying dataset sizes, run:

`python3 main.py --run_mode 4`

## Other Arguments

To adjust batch size, base learning rate for BYOL, weight decay for BYOL, epochs for BYOL, and input image size, you can specify the `bz`, `base_lr`, `weight_decay`, `byol_epochs`, and `image_size` arguments to `main.py`, respectively.


