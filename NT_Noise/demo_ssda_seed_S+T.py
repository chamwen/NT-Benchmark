# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import argparse
import os, sys
import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optim
from utils import network, utils
from utils.dataloader import read_seed_src_tar
from utils.utils import fix_random_seed, op_copy, lr_scheduler, data_load_noimg_ssda


def train_source_test_target(args):
    X_src, y_src, X_tar, y_tar = read_seed_src_tar(args)
    dset_loaders = data_load_noimg_ssda(X_src, y_src, X_tar, y_tar, args)

    netF, netC = network.backbone_net(args, args.bottleneck)
    netF.load_state_dict(tr.load(args.mdl_init_dir + 'netF.pt'))
    netC.load_state_dict(tr.load(args.mdl_init_dir + 'netC.pt'))

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 0.1}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    args.max_iter = max_iter
    iter_num = 0

    netF.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        try:
            inputs_target_tr, labels_target_tr = iter_target_tr.next()
        except:
            iter_target_tr = iter(dset_loaders["target_tr"])
            inputs_target_tr, labels_target_tr = iter_target_tr.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        inputs_target_tr, labels_target_tr = inputs_target_tr.cuda(), labels_target_tr.cuda()

        inputs_data = tr.cat((inputs_source, inputs_target_tr), 0)
        inputs_label = tr.cat((labels_source, labels_target_tr), 0)

        feas, output = netC(netF(inputs_data))
        classifier_loss = nn.CrossEntropyLoss()(output, inputs_label)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % (interval_iter * 2) == 0 or iter_num == max_iter:
            netF.eval()
            netC.eval()

            acc_s_te, _ = utils.cal_acc_noimg(dset_loaders["source_te"], netF, netC)
            acc_t_te, _ = utils.cal_acc_noimg(dset_loaders["Target"], netF, netC)
            log_str = 'Task: {}, Iter:{}/{}; Val_Acc = {:.2f}%; Test_Acc = {:.2f}%'.format(
                args.task_str, iter_num, max_iter, acc_s_te, acc_t_te)
            print(log_str)
            netF.train()
            netC.train()

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                acc_tar_src_best = acc_t_te
                netF.cuda()
                netC.cuda()

            netF.train()
            netC.train()

    return acc_tar_src_best


if __name__ == "__main__":
    data_name = 'SEED'
    if data_name == 'SEED': chn, class_num, trial_num = 62, 3, 3394
    focus_domain_idx = [0, 1, 2]
    # focus_domain_idx = np.arange(15)
    domain_list = ['S' + str(i) for i in focus_domain_idx]
    num_domain = len(domain_list)

    args = argparse.Namespace(bottleneck=64, lr=0.01, lr_decay1=0.1, lr_decay2=1.0,
                              epsilon=1e-05, layer='wn', smooth=0, is_save=False,
                              N=num_domain, chn=chn, trial=trial_num, class_num=class_num)

    args.dset = data_name
    args.method = 'ST'
    args.backbone = 'ShallowNet'
    args.batch_size = 32
    args.max_epoch = 50
    args.input_dim = 310
    args.norm = 'zscore'
    args.bz_tar_tr = int(args.batch_size / 2)
    args.bz_tar_te = args.batch_size
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'
    args.tar_lbl_rate = 5  # [5, 10, ..., 50]/100

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args.data_env = 'gpu'  # 'local'
    args.seed = 2022
    fix_random_seed(args.seed)
    tr.backends.cudnn.deterministic = True

    noise_list = np.linspace(0, 100, 11).tolist()
    num_test = len(noise_list)
    acc_all = np.zeros(num_test)
    s, t = 0, 1
    for ns in range(num_test):
        args.noise_rate = np.round(noise_list[ns] / 100, 2)
        dset_n = args.dset + '_' + str(args.noise_rate)
        print(dset_n, args.method)
        info_str = '\nnoise %s: %s --> %s' % (str(noise_list[ns]), domain_list[s], domain_list[t])
        print(info_str)
        args.src, args.tar = focus_domain_idx[s], focus_domain_idx[t]
        args.task_str = domain_list[s] + '_' + domain_list[t]
        print(args)

        acc_all[ns] = train_source_test_target(args)
    print('\nSub acc: ', np.round(acc_all, 2))
    print('Avg acc: ', np.round(np.mean(acc_all), 2))

    acc_sub_str = str(np.round(acc_all, 2).tolist())
    acc_mean_str = str(np.round(np.mean(acc_all), 2).tolist())

