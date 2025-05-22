import torch
import json
import joblib
import pickle
import torch.utils.data as data_utils
import numpy as np
import scipy.sparse as sp
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
# from model.ctr.inputs import *

tqdm.pandas()


def sample(df):
    positive_df = df[df.click.isin([1])]
    negative_df = df[df.click.isin([0])]
    del df

    # negative sample number >> positive sample number
    # downsample to twice of postive sampel number
    negative_df = negative_df.sample(n = len(positive_df) * 2)

    df = pd.concat([positive_df, negative_df])
    del positive_df, negative_df

    # shuffle
    df = df.sample(frac=1) 
    return df


def read_mtl_data(path=None, args=None):
    if not path:
        raise ValueError('The path of dataset can NOT be empty')

    # hist 1-10 record the history behavior of user
    df = pd.read_csv(path, usecols=["user_id", "item_id", "click", "like", "video_category", "gender", "age", "hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"])
    
    df['video_category'] = df['video_category'].astype(str)
    df = sample(df)

    if args.mtl_task_num == 2:
        label_columns = ['click', 'like']
        categorical_columns = ["user_id", "item_id", "video_category", "gender", "age", "hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"]
    elif args.mtl_task_num == 1:
        label_columns = ['click']
        categorical_columns = ["user_id", "item_id", "video_category", "gender", "age", "hist_1", "hist_2",
                               "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"]
    else:
        label_columns = ['like']
        categorical_columns = ["user_id", "item_id", "video_category", "gender", "age", "hist_1", "hist_2",
                               "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"]
    
    for col in tqdm(categorical_columns):
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])    # map label to int

    new_columns = categorical_columns + label_columns
    df = df.reindex(columns=new_columns)

    user_columns = ["user_id", "gender", "age"]
    user_feature_dict, item_feature_dict = {}, {}

    # recode the unique value numbers and index of each user/item column
    for idx, col in tqdm(enumerate(df.columns)):
        if col not in label_columns:
            if col in user_columns:
                user_feature_dict[col] = (len(df[col].unique()), idx)
            else:
                item_feature_dict[col] = (len(df[col].unique()), idx)

    # train set: 80%  val set: 10%  test set : 10%
    df = df.sample(frac=1)
    train_len = int(len(df) * 0.8)
    train_df = df[:train_len]

    tmp_df = df[train_len:]
    val_df = tmp_df[:int(len(tmp_df)/2)]
    test_df = tmp_df[int(len(tmp_df)/2):]

    return train_df, val_df, test_df, user_feature_dict, item_feature_dict


class mtlDataSet(data_utils.Dataset):
    def __init__(self, data, args):
        self.feature = data[0]
        self.args = args
        if args.mtl_task_num == 2:
            self.label1 = data[1]
            self.label2 = data[2]
        else:
            self.label = data[1]

    def __getitem__(self, index):
        feature = self.feature[index]
        if self.args.mtl_task_num == 2:
            label1 = self.label1[index]
            label2 = self.label2[index]
            return feature, label1, label2
        else:
            label = self.label[index]
            return feature, label

    def __len__(self):
        return len(self.feature)


def get_train_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.train_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
    return dataloader

def get_val_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.val_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True)
    return dataloader

def get_test_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.test_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
    return dataloader

