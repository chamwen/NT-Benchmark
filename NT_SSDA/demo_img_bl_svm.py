# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import numpy as np
from sklearn.metrics import accuracy_score
from utils.utils_bl import baseline_KNN, baseline_SVM

dset = 'DomainNet'
noise_rate = 0
dset_n = dset + '_' + str(noise_rate)
tar_lbl_rate = 5  # [5, 10, ..., 50]/100

if dset == 'DomainNet':
    domain_list = ['clipart', 'infograph', 'painting']
num_domain = len(domain_list)

acc_all = np.zeros(num_domain * (num_domain - 1))
for s in range(num_domain):
    for t in range(num_domain):
        if t == s:
            continue
        itr_idx = (num_domain - 1) * s + t
        if t > s: itr_idx -= 1
        info_str = '\n%s: %s --> %s' % (itr_idx, domain_list[s], domain_list[t])
        print(info_str)
        name_src = domain_list[s][0].upper()
        name_tar = domain_list[t][0].upper()
        task_str = name_src + name_tar

        # load pre-trained features:
        root_path = 'outputs/feas/'
        data_path = dset_n + '/' + 'tr' + str(tar_lbl_rate) + '/' + task_str + '_0.npz'
        data_dir = root_path + data_path
        data = np.load(data_dir)

        X_source, Y_source = data['X_source'], data['y_source']
        X_target_tr, Y_target_tr = data['X_target_tr'], data['y_target_tr']
        X_target_te, Y_target_te = data['X_target_te'], data['y_target_te']
        Xs = np.concatenate((X_source, X_target_tr), 0)
        Ys = np.concatenate((Y_source, Y_target_tr), 0)
        print(Xs.shape, Ys.shape, X_target_te.shape, Y_target_te.shape)

        # test SVM:
        result_SVM = baseline_SVM(Xs, Ys, X_target_te, Y_target_te)
        acc_all[itr_idx] = accuracy_score(Y_target_te, result_SVM) * 100
        print('SVM: {:.2f}'.format(acc_all[itr_idx]))

print('\ndone')
print('All acc: ', np.round(acc_all, 2))
print('Avg acc: ', np.round(np.mean(acc_all), 2))

