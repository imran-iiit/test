# Importing pytorch packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
# Importing config.py file
import config as cf
from utils import *
## Importing python packages
import os
import sys
import time
import datetime
import numpy as np
import math
import matplotlib.pyplot as plt

from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


def get_image_paths():
    img_root = 'signs_data'
    
# def load_images():
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'signs_data'
for turn in ['turn_left', 'misc', 'right_turns', 'stop', 'traffic_light', 'u_turn']:
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                        for x in ['left_turns', 'misc', 'right_turns', 'stop', 'traffic_light', 'u_turn']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                        for x in ['left_turns', 'misc', 'right_turns', 'stop', 'traffic_light', 'u_turn']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['left_turns', 'misc', 'right_turns', 'stop', 'traffic_light', 'u_turn']}
# class_names = image_datasets['images'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['images']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

# 
# def load_images():
#     img_root = cf.data_dir+'IMFDB_final/'
# 
#     train_list_file = cf.data_dir+'IMFDB_train.txt'   #### 5000 images for training
#     val_list_file = cf.data_dir+'IMFDB_test.txt'      #### 1095 images for validation
#     
#     
#     train_image_list = [line.rstrip('\n') for line in open(train_list_file)]
#     val_image_list = [line.rstrip('\n') for line in open(val_list_file)]
#     
#     print len(train_image_list), len(val_image_list)
#     
#     trainloader = torch.utils.data.DataLoader(custom_data_loader(img_root = img_root, image_list = train_list_file, crop=False,
#                                                                  resize = True, resize_shape=[128,128]), 
#                                                batch_size=32, num_workers=16, shuffle = True, pin_memory=True)
#     
#     testloader = torch.utils.data.DataLoader(custom_data_loader(img_root = img_root, image_list = val_list_file, crop=False, mirror=False, 
#                                                                resize = True, resize_shape=[128,128]), 
#                                                batch_size=10, num_workers=5, shuffle = False, pin_memory=True)