import os
import dload
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import Dataset

class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        
        self.n_pool = len(X_train)
        self.n_test = len(X_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        print(self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])[0])
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs])
    
    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)
        
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)
    
    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

    
def get_MNIST(handler):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_UnbalancedMNIST(handler):
    raw_train = datasets.MNIST('./data/MNIST', train=True, download=True)
    raw_test = datasets.MNIST('./data/MNIST', train=False, download=True)
    X_train = raw_train.data
    Y_train = raw_train.targets
    X_train = torch.cat((X_train[(Y_train==1)][:5000], X_train[(Y_train==0)][:1000]), 0)
    Y_train = torch.cat((Y_train[(Y_train == 1)][:5000], Y_train[(Y_train == 0)][:1000]), 0)
    X_test = torch.cat((raw_test.data[((raw_test.targets ==1))], raw_test.data[((raw_test.targets ==0))]), 0)
    Y_test = torch.cat((raw_test.targets[((raw_test.targets == 1))], raw_test.targets[((raw_test.targets == 0))]), 0)
    return Data(X_train, Y_train, X_test, Y_test, handler)

def get_FashionMNIST(handler):
    raw_train = datasets.FashionMNIST('./data/FashionMNIST', train=True, download=True)
    raw_test = datasets.FashionMNIST('./data/FashionMNIST', train=False, download=True)
    return Data(raw_train.data[:40000], raw_train.targets[:40000], raw_test.data[:40000], raw_test.targets[:40000], handler)

def get_SVHN(handler):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data[:40000], torch.from_numpy(data_train.labels)[:40000], data_test.data[:40000], torch.from_numpy(data_test.labels)[:40000], handler)

def get_CIFAR10(handler):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data[:40000], torch.LongTensor(data_train.targets)[:40000], data_test.data[:40000], torch.LongTensor(data_test.targets)[:40000], handler)

"""
extra functions for nlp data creation
"""

class TextDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data, self.targets = self._load_data()

    def __getitem__(self, index):
        text = self.data[index]
        label = self.targets[index]
        return text, label

    def __len__(self):
        return len(self.data)

    def _load_data(self):
        data = []
        targets = []
        for label in sorted(os.listdir(self.root_dir)):
            path = os.path.join(self.root_dir, label)
            if os.path.isdir(path):
                for fname in sorted(os.listdir(path)):
                    fpath = os.path.join(path, fname)
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data.append(f.read())
                    targets.append(int(0 if label == "neg" else 1))
        return data, targets


def prepareData(root_dir):
    dataset = TextDataset(root_dir)
    return dataset

def get_MovieReview(handler):
    dload.save_unzip("https://victorzhou.com/movie-reviews-dataset.zip","./data")
    data_train = prepareData('./data/movie-reviews-dataset/train')
    data_test = prepareData('./data/movie-reviews-dataset/test')
    print(type(data_train.data))
    print(type(data_train.targets))

    return Data(torch.as_tensor(data_train.data), torch.as_tensor(data_train.targets), torch.as_tensor(data_test.data), torch.as_tensor(data_test.targets), handler)