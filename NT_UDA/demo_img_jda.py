# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import numpy as np
from utils.utils_bl import JDA

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

        # load pre-trained features:
        root_path = 'outputs/feas/'
        data_path = dset_n + '/' + task_str + '_0.npz'
        data_dir = root_path + data_path
        data = np.load(data_dir)
        Xs, Ys = data['X_source'], data['y_source']
        Xt, Yt = data['X_target'], data['y_target']
        print(Xs.shape, Ys.shape, Xt.shape, Yt.shape)

        # JDA
        ker_type = 'primal'
        traditional_tl = JDA(kernel_type=ker_type, dim=100, lamb=1, gamma=1)
        acc_all[itr_idx] = traditional_tl.fit_predict(Xs, Ys, Xt, Yt)
        print('JDA: {:.2f}'.format(acc_all[itr_idx]))

print('\ndone')
print('All acc: ', np.round(acc_all, 2))
print('Avg acc: ', np.round(np.mean(acc_all), 2))

