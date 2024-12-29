import timeit
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import transforms, datasets,tv_tensors,models
from torchvision.transforms import v2

import numpy as np
import random

#Function for reproducibilty. You can check out: https://pytorch.org/docs/stable/notes/randomness.html
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(100)

#TODO: Populate the dictionary with your hyperparameters for training
def get_config_dict(pretrain):
    """
    pretrain: 0 or 1. Can be used if you need different configs for part 1 and 2.
    """
    if pretrain == 0:
        config = {
            "batch_size": 128,
            "lr": 0.001,
            "num_epochs": 10,
            "weight_decay": 0,   #set to 0 if you do not want L2 regularization
            "save_criteria": "accuracy",     #Str. Can be 'accuracy'/'loss'/'last'. (Only for part 2)
        }
    else:
        config = {
            "batch_size": 128,
            "lr": 0.00001,
            "num_epochs": 50,
            "weight_decay": 0,   #set to 0 if you do not want L2 regularization
            "save_criteria": "accuracy",     #Str. Can be 'accuracy'/'loss'/'last'. (Only for part 2)

        }
    
    
    return config
    

#TODO: Part 1 - Complete this with your CNN architecture. Make sure to complete the architecture requirements.
class Net(nn.Module):
    """
    Description: A simple CNN architecture for CIFAR-10 classification.
    """
    def __init__(self):
        super(Net, self).__init__()
        # Conv Layer 1
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 5, padding = 1)
        self.relu1 = nn.ReLU()
        # Max Pooling 1
        self.pool1 = nn.MaxPool2d(3, 2)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, padding = 1)
        self.relu2 = nn.ReLU()
        # Max Pooling 2
        self.pool2 = nn.MaxPool2d(3,2)

        # Conv Layer 3
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 1)
        self.relu3 = nn.ReLU()
        # Max Pooling 3
        self.pool3 = nn.MaxPool2d(3, 2)

        # Fully Connected Layer 1
        self.fc1 = nn.Linear(64, 120)



    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.flatten(1)
        x = self.fc1(x)
        return x


#TODO: Part 2 - Complete this with your Pretrained CNN architecture. 
class PretrainedNet(nn.Module):
    """
    Description: A CNN architecture using a pretrained model for CIFAR-10 classification.
    """
    def __init__(self):
        super(PretrainedNet, self).__init__()
        # TODO: Load a pretrained model
        # using the convnext_tiny model
        self.model = models.convnext_tiny(pretrained=True)  # Load the pretrained model
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, 10)  # Change the output layer to have 10 classes
        print("Model summary:",self.model)

    def forward(self, x):
        return self.model(x)


#Feel free to edit this with your custom train/validation splits, transformations and augmentations for CIFAR-10, if needed.
def load_dataset(pretrain):
    """
    pretrain: 0 or 1. Can be used if you need to define different dataset splits/transformations/augmentations for part 2.

    returns:
    train_dataset, valid_dataset: Dataset for training your model
    test_transforms: Default is None. Edit if you would like transformations applied to the test set. 

    """
    if pretrain == 0:
            full_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                    transform=v2.Compose([
                                        v2.ToTensor(),
                                        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    else:
        transforms = v2.Compose([
                v2.RandomCrop(32, padding=4),  # Random cropping
                v2.RandomHorizontalFlip(),     # Horizontal flip
                v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1), # Color jitter
                v2.RandomRotation(15),         # Random rotation
                v2.ToTensor(),                # Convert to tensor   
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize
        ])
        full_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                    transform=transforms) 

    train_dataset, valid_dataset = random_split(full_dataset, [38000, 12000])

    test_transforms = None

    
    return train_dataset, valid_dataset, test_transforms




