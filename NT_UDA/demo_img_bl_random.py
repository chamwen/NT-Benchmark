# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import numpy as np
from sklearn.metrics import accuracy_score

dset = 'DomainNet'
noise_rate = 0
dset_n = dset + '_' + str(noise_rate)

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

        # load labels
        folder = "checkpoint/"
        s_dset_path = folder + dset + '/' + domain_list[s] + '_list.txt'
        t_dset_path = folder + dset + '/' + domain_list[t] + '_list.txt'
        txt_src = open(s_dset_path).readlines()
        txt_tar = open(t_dset_path).readlines()
        Y_source = [int(img_str.split('==')[1]) for img_str in txt_src]
        Y_target = [int(img_str.split('==')[1]) for img_str in txt_tar]

        # random guess
        class_list = np.unique(Y_source)
        class_num_list = [np.sum(Y_source == c) for c in class_list]
        num_max_class = class_list[np.argmax(class_num_list)]
        pred_tar = np.ones(len(Y_target)) * num_max_class
        pred_tar = pred_tar.astype(int)
        acc_all[itr_idx] = accuracy_score(Y_target, pred_tar) * 100

        print('acc: {:.2f}'.format(acc_all[itr_idx]))

print('\ndone')
print('All acc: ', np.round(acc_all, 2))
print('Avg acc: ', np.round(np.mean(acc_all), 2))

