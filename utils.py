import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torchvision.models import resnet18
import random
from kornia import augmentation as aug
from kornia import filters
from kornia.geometry import transform as tf
import numpy as np
from copy import deepcopy

def accuracy(pred, labels):
    return (pred.argmax(dim=-1) == labels).float().mean().item()

def byol_loss_fn(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 2 - 2 * (x * y).sum(dim=-1)

def get_resnet():
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(512, 10)
    return model

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x if random.random() > self.p else self.fn(x)

def supervised_augmentation():
    return nn.Sequential(
        aug.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
    )

def default_augmentation(image_size=(224, 224)):
    return nn.Sequential(
        tf.Resize(size=image_size),
        RandomApply(aug.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
        aug.RandomGrayscale(p=0.2),
        aug.RandomHorizontalFlip(),
        RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
        aug.RandomResizedCrop(size=image_size),
        #RandomPatchDrop(size=image_size),
        #aug.RandomPerspective(),
        aug.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
    )


def image_transform(image_size):
    return nn.Sequential(
        tf.Resize(size=image_size),
        aug.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
    )
    

class RandomPatchDrop(torch.nn.Module):
    def __init__(self, size=(224,224), num_patches=3, patch_size_range=(0.1, 0.3)):
        super().__init__()
        self.image_size = size
        self.patch_size_range = patch_size_range
        self.num_patches = num_patches
    
    def forward(self, img):
        img = deepcopy(img)
        for b in range(img.shape[0]):
            for p in range(self.num_patches):
                patch_size_x = np.random.uniform(self.patch_size_range[0], self.patch_size_range[1])
                patch_size_x = np.round(patch_size_x * self.image_size[0])
                x1 = int(self.image_size[0]-patch_size_x)
                x1 = np.random.choice(np.arange(x1))

                patch_size_y = np.random.uniform(self.patch_size_range[0], self.patch_size_range[1])
                patch_size_y = np.round(patch_size_y * self.image_size[1])
                y1 = int(self.image_size[1]-patch_size_y)
                y1 = np.random.choice(np.arange(y1))
                img = deepcopy(img)
                img[b,:,y1:int(y1+patch_size_y),x1:int(x1+patch_size_x)]=0
        return img