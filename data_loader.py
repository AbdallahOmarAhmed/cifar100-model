import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import pickle

def unpickle(path):
    with open(path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class cifar100DataSet(Dataset):
    path = '/home/abdallah/cifar-100-python/train'
    def __init__(self):
        self.data = unpickle(cifar100DataSet.path)
        self.images=self.data[b'data'].reshape(-1, 3, 32, 32)
        self.labels = self.data[b'coarse_labels']
        self.data_set_size = self.images.shape[0]
    def __len__(self):
        return self.data_set_size
    def __getitem__(self,index):        
        return torch.from_numpy(self.images[index]).to(torch.float32)/255, torch.tensor(self.labels[index]).to(torch.long)

class cifar100TestSet(Dataset):
    path = '/home/abdallah/cifar-100-python/test'
    def __init__(self):
        self.data = unpickle(cifar100TestSet.path)
        self.images=self.data[b'data'].reshape(-1, 3, 32, 32)
        self.labels = self.data[b'coarse_labels']
        self.data_set_size = self.images.shape[0]    
    def __len__(self):
        return self.data_set_size
    def __getitem__(self,index):        
        return torch.from_numpy(self.images[index]).to(torch.float32)/255, torch.tensor(self.labels[index]).to(torch.long)


