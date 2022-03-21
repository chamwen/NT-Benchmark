# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from utils.utils import add_label_noise_noimg
from utils.utils_bl import KMM, baseline_SVM

data_name = 'moon'
seed = 2022
if data_name == 'moon': class_num = 2
noise_rate = 0
base_name_list = ['0', '1', '2', '3_45', '4_15', '6', '7', '8', '9']
domain_list = ['Raw', 'Tl', 'Sl', 'Rt', 'Sh', 'Sk', 'Ns', 'Ol', 'Sc']
file_list = [data_name + i for i in base_name_list]
num_domain = len(domain_list)

root_path = './data_synth/'
acc_all = np.zeros((len(domain_list) - 1))
for s in range(1, num_domain):  # source
    for t in [0]:  # target
        itr_idx = s - 1
        print('\n%s: %s --> %s' % (itr_idx, domain_list[s], domain_list[t]))
        src, tar = file_list[s], file_list[t]
        pd_src = pd.read_csv(root_path + src + ".csv", header=None)
        Xs, Ys = pd_src.iloc[:, :2].values, pd_src.iloc[:, 2].values.astype(int)
        pd_tar = pd.read_csv(root_path + tar + ".csv", header=None)
        Xt, Yt = pd_tar.iloc[:, :2].values, pd_tar.iloc[:, 2].values.astype(int)

        # add noise on source label
        Ys = add_label_noise_noimg(Ys, seed, class_num, noise_rate)

        kmm = KMM(kernel_type='rbf', B=1)
        beta = kmm.fit(Xs, Xt)
        Xs_new = beta * Xs

        pred_tar = baseline_SVM(Xs_new, Ys, Xt, Yt)
        acc_all[itr_idx] = accuracy_score(Yt, pred_tar) * 100
        print('KMM: %.2f' % np.round(acc_all[itr_idx], 2))

print('\nmean acc', np.round(np.mean(acc_all), 2))
print(domain_list)
print(np.round(acc_all, 2).tolist())

