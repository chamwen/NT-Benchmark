# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import numpy as np
import torch as tr
import argparse
from utils.utils import add_label_noise_noimg
from utils.dataloader import data_normalize
from utils.utils_bl import JDA


def read_seed_src_tar_bl(args):
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
# focus_domain_idx = np.arange(15)
domain_list = ['S' + str(i) for i in focus_domain_idx]
num_domain = len(domain_list)

args = argparse.Namespace(dset=data_name, norm='zscore', seed=2022, class_num=3)
args.data_env = 'gpu'  # 'local'
args.noise_rate = 0

num_domain = len(domain_list)
acc_all = np.zeros(len(domain_list) * (len(domain_list) - 1))
for s in range(num_domain):  # source
    for t in range(num_domain):  # target
        if s != t:
            itr_idx = (num_domain - 1) * s + t
            if t > s: itr_idx -= 1
            print('\n%s: %s --> %s' % (itr_idx, domain_list[s], domain_list[t]))
            args.src, args.tar = s, t
            Xs, Ys, Xt, Yt = read_seed_src_tar_bl(args)

            # add noise on source label
            Ys = add_label_noise_noimg(Ys, args.seed, args.class_num, args.noise_rate)

            # JDA
            ker_type = 'primal'
            traditional_tl = JDA(kernel_type=ker_type, dim=100, lamb=1, gamma=1)
            acc_all[itr_idx] = traditional_tl.fit_predict(Xs, Ys, Xt, Yt)
            print('JDA: {:.2f}'.format(acc_all[itr_idx]))

print('\nmean acc', np.round(np.mean(acc_all), 2))
print(domain_list)
print(np.round(acc_all, 2).tolist())

