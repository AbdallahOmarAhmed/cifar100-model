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

class myDataSet(Dataset):
    path = '/home/abdallah/cifar-10-batches-py/data_batch_'
    def __init__(self):
        self.batch1 = unpickle(myDataSet.path + '1')
        self.images1=self.batch1[b'data'].reshape(-1, 3, 32, 32)
        self.labels1 = self.batch1[b'labels']

        self.batch2 = unpickle(myDataSet.path + '2')
        self.images2=self.batch2[b'data'].reshape(-1, 3, 32, 32)
        self.labels2 = self.batch2[b'labels']

        self.batch3 = unpickle(myDataSet.path + '3')
        self.images3=self.batch3[b'data'].reshape(-1, 3, 32, 32)
        self.labels3 = self.batch3[b'labels']

        self.batch4 = unpickle(myDataSet.path + '4')
        self.images4=self.batch4[b'data'].reshape(-1, 3, 32, 32)
        self.labels4 = self.batch4[b'labels']

        self.batch5 = unpickle(myDataSet.path + '5')
        self.images5=self.batch5[b'data'].reshape(-1, 3, 32, 32)
        self.labels5 = self.batch5[b'labels']

        self.X = np.concatenate([self.images1, self.images2, self.images3, self.images4, self.images5], axis=0)
        self.Y = self.labels1 + self.labels2 + self.labels3 + self.labels4 + self.labels5
        #self.test_batch = unpickle('/home/abdallah/cifar-10-batches-py/test_batch')

        self.input_size = self.X.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]).to(torch.float32)/255, torch.tensor(self.Y[index]).to(torch.long)

    def __len__(self):
        return self.input_size

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


