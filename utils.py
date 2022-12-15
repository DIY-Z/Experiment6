import torch
import torchvision.transforms as trans
from torch.utils.data import Dataset, DataLoader
import gzip
import os
import numpy as np
import copy

local_file = '/data/HomeWork/Experiment6/data'
files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

class CustomDataset(Dataset):
    def __init__(self, data):
        self.img_list = copy.deepcopy(data[0])
        self.label_list = copy.deepcopy(data[1])
        
        self.transform = trans.Compose([trans.ToTensor()])


    def __getitem__(self, index):
        img = self.img_list[index]
        img = self.transform(img)
        label = self.label_list[index]
        return (img, label)

    def __len__(self):
        return len(self.img_list)

def load_local_mnist(filename):# 加载文件
    paths = []
    file_read = []
    for file in files:
        paths.append(os.path.join(filename, file))
    for path in paths:
        file_read.append(gzip.open(path, 'rb'))
    # print(file_read)

    train_labels = np.frombuffer(file_read[1].read(), np.uint8, offset=8)#文件读取以及格式转换
    train_images = np.frombuffer(file_read[0].read(), np.uint8, offset=16).reshape(len(train_labels), 28, 28)
    test_labels = np.frombuffer(file_read[3].read(), np.uint8, offset=8)
    test_images = np.frombuffer(file_read[2].read(), np.uint8, offset=16).reshape(len(test_labels), 28, 28)
    return (train_images, train_labels), (test_images, test_labels)

def get_dataloader():
        #加载训练集、测试集数据及标签
    (X_train, y_train), (X_test, y_test) = load_local_mnist(local_file)

    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape) #(60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
    
    # X_train = np.expand_dims(X_train, 1)
    # X_test = np.expand_dims(X_test, 1)

    train_dataset = CustomDataset((X_train, y_train))
    test_dataset = CustomDataset((X_test, y_test))
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=16
    )

    return train_dataloader, test_dataloader