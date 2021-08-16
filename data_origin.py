from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import SubsetRandomSampler


def read_data(data_path):
    df = pd.read_csv(data_path)
    # print(df.shape)
    # df.drop_duplicates('entid', 'first', inplace=True)
    # print(df.shape)
    y = df['CaseType']
    X = df.drop('entid', axis=1).drop('Inconfidence', axis=1).drop('ANCHEYEAR', axis=1).drop('ANCHEDATE', axis=1).drop(
        'Year', axis=1).drop('CaseType', axis=1).drop('debt', axis=1)
    X = X.fillna(0)
    X = StandardScaler().fit_transform(X)
    return X, y


def data_split(data_path, batch_size, shuffle_dataset=True, random_seed=42, val_p=0.2, train_balance=True):
    dataset = FeatureDataSet(data_path)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_p * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    return train_loader, validation_loader


class FeatureDataSet(Dataset):
    def __init__(self, data_path):
        self.data, self.label = read_data(data_path)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)