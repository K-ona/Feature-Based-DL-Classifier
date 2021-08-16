from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import SubsetRandomSampler
from collections import defaultdict

drop_list = [
    'entid', 'Inconfidence', 'ANCHEYEAR', 'ANCHEDATE', 'Year', 'CaseType',
    # 'debt', 'Social', 'Ratio', 'CONAM_new', 'ttype', 'assets_re', 'INDUSTRYPHY'
]

type_cnt = defaultdict(list)


def sub_year(df):
    df['sub_year'] = ''
    sz = df.shape[0]
    for i in range(sz):
        df['ANCHEDATE'][i] = str(df['ANCHEDATE'][i])[:4]
    df['ANCHEDATE'] = df['ANCHEDATE'].astype(int)
    df['sub_year'] = df['ANCHEYEAR'] - df['ANCHEDATE']
    return df


def read_data(data_path):
    df = pd.read_csv(data_path, encoding='GBK')
    y = df['CaseType']
    # df = sub_year(df)
    for item in drop_list:
        df = df.drop(item, axis=1)

    for num in range(int(y.max()) + 1):
        type_cnt[num] = df[y == num].index

    X = df.fillna(0)
    X = StandardScaler().fit_transform(X)
    return X, y


def read_test_data(data_path):
    df = pd.read_csv(data_path)
    ids = df['entid']
    cor_dict = defaultdict(list)
    for i, v in enumerate(ids):
        cor_dict[v].append(i)

    for item in drop_list:
        df = df.drop(item, axis=1)

    x = df.fillna(0)
    return StandardScaler().fit_transform(x), ids, cor_dict


def make_dataset(data_path, res_file):
    df = pd.read_csv(data_path)
    origin_dist = df['CaseType'].astype(int)
    cnt = defaultdict(int)
    for num in range(5):
        cnt[num] = (origin_dist == num).sum()
    Max_size = max(cnt.values())
    for num in range(4):
        scale = int(Max_size / cnt[num]) - 1
        # print('scale == ', scale)
        df = df.append([df[df['CaseType'] == num]] * scale)
        # print(df.shape)
    df.to_csv(res_file, index=None)


def balance_data(train_indices, random_seed):
    res = []
    Max_cnt = max([len(l) for l in type_cnt.values()])
    # Scale = [int(Max_cnt / len(x)) for x in type_cnt.values()]
    Scale = [100, 10, 3, 1, 1]

    for num in range(max(type_cnt.keys()) + 1):
        tmp = []
        for ind in train_indices:
            if ind in type_cnt[num]:
                tmp.append(ind)
        if isinstance(Scale[num], int):
            res += tmp * Scale[num]
        else:
            res += func(Scale[num], tmp, random_seed)
            # if Scale[num] is int else func(Scale[num], tmp)
    return res


def func(selectp, List, random_seed):
    split = int(selectp * len(List))
    np.random.seed(random_seed)
    np.random.shuffle(List)
    return List[:split]


def data_split(data_path, batch_size, shuffle_dataset=True, random_seed=42, val_p=0.2, train_balance=True):
    dataset = FeatureDataSet(data_path)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_p * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # cnt = defaultdict(int)
    # for ind in train_indices:
    #     cnt[ind] += 1

    if train_balance:
        train_indices = balance_data(train_indices, random_seed)

    # cnt = defaultdict(int)
    # for ind in train_indices:
    #     cnt[ind] += 1

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    return train_loader, validation_loader


def get_dataset(train_path, val_path, batch_size):
    traindata = FeatureDataSet(train_path)
    valdata = FeatureDataSet(val_path)
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(valdata, batch_size=batch_size, shuffle=True)
    return train_loader, validation_loader


class FeatureDataSet(Dataset):
    def __init__(self, data_path):
        self.data, self.label = read_data(data_path)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
