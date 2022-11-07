# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import os
import numpy as np
import torch as tr
import argparse
from sklearn.metrics import accuracy_score
from utils.utils import add_label_noise_noimg, get_idx_ssda_seed
from utils.dataloader import data_normalize
from utils.utils_bl import baseline_SVM


def dataload(args):
    # (15, 3394, 310) (15, 3394)
    if args.data_env == 'local':
        file = 'D:/Dataset/MOABB/' + args.dset + '.npz'
    if args.data_env == 'gpu':
        file = '/mnt/ssd2/wenz/data/bci/' + args.dset + '.npz'

    MI = np.load(file)
    Data_raw, Label = MI['data'], MI['label']

    src_data = np.squeeze(Data_raw[args.src, :, :])
    src_data = data_normalize(src_data, args.norm)
    src_label = np.squeeze(Label[args.src, :])

    # target sub
    tar_data = np.squeeze(Data_raw[args.tar, :, :])
    tar_data = data_normalize(tar_data, args.norm)
    tar_label = np.squeeze(Label[args.tar, :])
    print(tar_data.shape, tar_label.shape)

    return src_data, src_label, tar_data, tar_label


data_name = 'SEED'
if data_name == 'SEED': chn, class_num, trial_num = 62, 3, 3394
focus_domain_idx = [0, 1, 2]
domain_list = ['S' + str(i) for i in focus_domain_idx]
num_domain = len(domain_list)

args = argparse.Namespace(dset=data_name, norm='zscore', seed=2022, class_num=3)
args.data_env = 'gpu'  # 'local'
args.noise_rate = 0
args.tar_lbl_rate = 5  # [5, 10, ..., 50]/100

num_domain = len(domain_list)
acc_all = np.zeros(len(domain_list) * (len(domain_list) - 1))
for s in range(num_domain):  # source
    for t in range(num_domain):  # target
        if s != t:
            itr_idx = (num_domain - 1) * s + t
            if t > s: itr_idx -= 1
            print('\n%s: %s --> %s' % (itr_idx, domain_list[s], domain_list[t]))
            args.src, args.tar = s, t
            Xs, Ys, Xt, Yt = dataload(args)

            idx_tar_tr, idx_tar_te = get_idx_ssda_seed(Yt, args.tar_lbl_rate)
            Xt_tr, Yt_tr = Xt[idx_tar_tr, :], Yt[idx_tar_tr]
            Xt_te, Yt_te = Xt[idx_tar_te, :], Yt[idx_tar_te]
            Xs = np.concatenate((Xs, Xt_tr), 0)
            Ys = np.concatenate((Ys, Yt_tr), 0)
            Xt, Yt = Xt_te.copy(), Yt_te.copy()
            # print(Xs.shape, Ys.shape, Xt_te.shape, Yt_te.shape)

            # add noise on source label
            Ys = add_label_noise_noimg(Ys, args.seed, args.class_num, args.noise_rate)

            # test SVM:
            pred_tar = baseline_SVM(Xs, Ys, Xt, Yt)
            acc_all[itr_idx] = accuracy_score(Yt, pred_tar) * 100
            print('acc: %.2f' % np.round(acc_all[itr_idx], 2))

print('\nmean acc', np.round(np.mean(acc_all), 2))
print(domain_list)
print(np.round(acc_all, 2).tolist())

