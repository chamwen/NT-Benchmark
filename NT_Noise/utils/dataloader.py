# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import pandas as pd
import torch as tr
from torch.autograd import Variable
import numpy as np
from sklearn import preprocessing


def read_syn_single(args, sub_idx):
    root_path = args.root_path
    pd_tar = pd.read_csv(root_path + sub_idx + ".csv", header=None)
    X, Y = pd_tar.iloc[:, :2].values, pd_tar.iloc[:, 2].values.astype(int)
    X = Variable(tr.from_numpy(X).float())
    Y = tr.from_numpy(Y).long()

    return X, Y


def read_syn_src_tar(args):
    root_path = args.root_path
    pd_src = pd.read_csv(root_path + args.src + ".csv", header=None)
    Xs, Ys = pd_src.iloc[:, :2].values, pd_src.iloc[:, 2].values.astype(int)
    pd_tar = pd.read_csv(root_path + args.tar + ".csv", header=None)
    Xt, Yt = pd_tar.iloc[:, :2].values, pd_tar.iloc[:, 2].values.astype(int)
    Xs = Variable(tr.from_numpy(Xs).float())
    Ys = tr.from_numpy(Ys).long()
    Xt = Variable(tr.from_numpy(Xt).float())
    Yt = tr.from_numpy(Yt).long()

    return Xs, Ys, Xt, Yt


def data_normalize(fea_de, norm_type):
    if norm_type == 'zscore':
        zscore = preprocessing.StandardScaler()
        fea_de = zscore.fit_transform(fea_de)
    return fea_de


def read_seed_single(args, sub_idx):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = 'D:/Dataset/MOABB/' + args.dset + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wenz/data/bci/' + args.dset + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    # source sub
    fea_de = np.squeeze(Data_raw[sub_idx, :, :])
    fea_de = data_normalize(fea_de, args.norm)
    fea_de = Variable(tr.from_numpy(fea_de).float())

    sub_label = np.squeeze(Label[sub_idx, :])
    sub_label = tr.from_numpy(sub_label).long()
    print(fea_de.shape, sub_label.shape)

    return fea_de, sub_label


def read_seed_src_tar(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = 'D:/Dataset/MOABB/' + args.dset + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wenz/data/bci/' + args.dset + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    src_data = np.squeeze(Data_raw[args.src, :, :])
    src_data = data_normalize(src_data, args.norm)
    src_data = Variable(tr.from_numpy(src_data).float())
    src_label = np.squeeze(Label[args.src, :])
    src_label = tr.from_numpy(src_label).long()

    # target sub
    tar_data = np.squeeze(Data_raw[args.tar, :, :])
    tar_data = data_normalize(tar_data, args.norm)
    tar_data = Variable(tr.from_numpy(tar_data).float())
    tar_label = np.squeeze(Label[args.tar, :])
    tar_label = tr.from_numpy(tar_label).long()
    print(tar_data.shape, tar_label.shape)

    return src_data, src_label, tar_data, tar_label


def obtain_train_val_source(y_array, trial_ins_num, val_type):
    y_array = y_array.numpy()
    ins_num_all = len(y_array)
    src_idx = range(ins_num_all)

    if val_type == 'random':
        num_train = int(0.9 * len(src_idx))
        id_train, id_val = tr.utils.data.random_split(src_idx, [num_train, len(src_idx) - num_train])

    return id_train, id_val
