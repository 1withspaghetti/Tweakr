import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LRScheduler
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import copy
from collections import namedtuple
import os
import random
import shutil
import time

# Set the random seed for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Define the root directory for the dataset
ROOT = 'data'

# Extract the dataset archive
datasets.utils.extract_archive('CUB_200_2011.tgz', ROOT)

# Define the ratio for splitting the data into training and testing sets
TRAIN_RATIO = 0.8

# Define directories for data
data_dir = os.path.join(ROOT, 'CUB_200_2011')
images_dir = os.path.join(data_dir, 'images')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Remove existing training and testing directories if they exist
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

# Create new training and testing directories
os.makedirs(train_dir)
os.makedirs(test_dir)

# List all classes (subdirectories) in the images directory
classes = os.listdir(images_dir)

# Iterate over each class to split images into training and testing sets
for c in classes:
    class_dir = os.path.join(images_dir, c)
    images = os.listdir(class_dir)

    # Calculate the number of training images
    n_train = int(len(images) * TRAIN_RATIO)

    # Split images into training and testing sets
    train_images = images[:n_train]
    test_images = images[n_train:]

    # Create class subdirectories in training and testing directories
    os.makedirs(os.path.join(train_dir, c), exist_ok=True)
    os.makedirs(os.path.join(test_dir, c), exist_ok=True)

    # Copy training images to the training directory
    for image in train_images:
        image_src = os.path.join(class_dir, image)
        image_dst = os.path.join(train_dir, c, image)
        shutil.copyfile(image_src, image_dst)

    # Copy testing images to the testing directory
    for image in test_images:
        image_src = os.path.join(class_dir, image)
        image_dst = os.path.join(test_dir, c, image)
        shutil.copyfile(image_src, image_dst)

# Create a dataset for calculating mean and standard deviation of images
train_data = datasets.ImageFolder(root=train_dir, transform=transforms.ToTensor())

# Initialize mean and standard deviation tensors
means = torch.zeros(3)
stds = torch.zeros(3)

# Calculate mean and standard deviation for each channel
for img, label in train_data:
    means += torch.mean(img, dim=(1, 2))
    stds += torch.std(img, dim=(1, 2))

# Average the mean and standard deviation by the number of images
means /= len(train_data)
stds /= len(train_data)

print(f'Calculated means: {means}')
print(f'Calculated stds: {stds}')

# Define the size and normalization parameters for pretrained models
pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds = [0.229, 0.224, 0.225]

# Define transforms for training data
train_transforms = transforms.Compose([
    transforms.Resize(pretrained_size),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(pretrained_size, padding=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
])

# Define transforms for testing data
test_transforms = transforms.Compose([
    transforms.Resize(pretrained_size),
    transforms.CenterCrop(pretrained_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
])

# Load datasets with applied transformations
train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)

# Define the validation split ratio
VALID_RATIO = 0.9

# Calculate the number of training and validation examples
n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

# Split the training data into training and validation sets
train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])

# Apply test transforms to the validation data
valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')