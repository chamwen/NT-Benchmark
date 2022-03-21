# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import argparse
import os, sys
import os.path as osp
import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from utils.LogRecord import LogRecord
from utils import network, utils
from utils.utils import fix_random_seed, op_copy, lr_scheduler, get_idx_ssda_seed
from utils.dataloader import read_seed_single


def data_load(X, y, args):
    dset_loaders = {}
    train_bs = args.batch_size

    idx_train, idx_test = get_idx_ssda_seed(y, args.tar_lbl_rate)

    data_tar_tr = Data.TensorDataset(X[idx_train, :], y[idx_train])
    data_tar_te = Data.TensorDataset(X[idx_test, :], y[idx_test])

    dset_loaders["target_tr"] = Data.DataLoader(data_tar_tr, batch_size=train_bs, shuffle=True)
    dset_loaders["Target"] = Data.DataLoader(data_tar_te, batch_size=train_bs * 3, shuffle=False)
    return dset_loaders


def train_source_test_target(args):
    X_tar, y_tar = read_seed_single(args, args.tar)
    dset_loaders = data_load(X_tar, y_tar, args)

    netF, netC = network.backbone_net(args, args.bottleneck)

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(tr.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(tr.load(modelpath))
    netF.eval()

    for k, v in netF.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netC.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * 0.1}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target_tr"])
    interval_iter = max_iter // 10
    args.max_iter = max_iter
    iter_num = 0
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_target_tr, labels_target_tr = iter_target_tr.next()
        except:
            iter_target_tr = iter(dset_loaders["target_tr"])
            inputs_target_tr, labels_target_tr = iter_target_tr.next()

        if inputs_target_tr.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_data, inputs_label = inputs_target_tr.cuda(), labels_target_tr.cuda()
        _, output = netC(netF(inputs_data))
        classifier_loss = nn.CrossEntropyLoss()(output, inputs_label)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % (interval_iter) == 0 or iter_num == max_iter:
            netC.eval()

            acc_t_te,_ = utils.cal_acc_noimg(dset_loaders["Target"], netF, netC)
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, iter_num, max_iter, acc_t_te)
            print(log_str)

            netC.train()

    return acc_t_te


if __name__ == "__main__":
    data_name = 'SEED'
    if data_name == 'SEED': chn, class_num, trial_num = 62, 3, 3394
    focus_domain_idx = [0, 1, 2]
    # focus_domain_idx = np.arange(15)
    domain_list = ['S' + str(i) for i in focus_domain_idx]
    num_domain = len(domain_list)

    args = argparse.Namespace(bottleneck=64, lr=0.01, lr_decay1=0.1,
                              epsilon=1e-05, layer='wn', smooth=0, is_save=False,
                              N=num_domain, chn=chn, trial=trial_num, class_num=class_num)

    args.dset = data_name
    args.method = 'Finetune'
    args.backbone = 'ShallowNet'
    args.batch_size = 32  # 32
    args.max_epoch = 50  # 50
    args.input_dim = 310
    args.norm = 'zscore'
    args.bz_tar_tr = int(args.batch_size / 2)
    args.bz_tar_te = args.batch_size
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'
    args.noise_rate = 0
    dset_n = args.dset + '_' + str(args.noise_rate)
    args.tar_lbl_rate = 5  # [5, 10, ..., 50]/100

    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    args.data_env = 'gpu'  # 'local'
    args.seed = 2022
    fix_random_seed(args.seed)
    tr.backends.cudnn.deterministic = True
    print(dset_n, args.method)
    print(args)

    mdl_path = 'outputs/models/'
    args.output = mdl_path + dset_n + '/source/'

    args.local_dir = r'/mnt/ssd2/wenz/NT-Benchmark/NT_SSDA/'
    args.result_dir = 'results/target/'
    my_log = LogRecord(args)
    my_log.log_init()
    my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

    acc_all = np.zeros(num_domain * (num_domain - 1))
    for s in range(num_domain):
        for t in range(num_domain):
            if s != t:
                itr_idx = (num_domain - 1) * s + t
                if t > s: itr_idx -= 1
                info_str = '\n%s: %s --> %s' % (itr_idx, domain_list[s], domain_list[t])
                print(info_str)
                args.src, args.tar = focus_domain_idx[s], focus_domain_idx[t]
                args.task_str = domain_list[s] + '_' + domain_list[t]

                args.name_src = domain_list[s]
                args.output_dir_src = osp.join(args.output, args.name_src)
                print(args)

                my_log.record(info_str)
                args.log = my_log

                acc_all[itr_idx] = train_source_test_target(args)
    print('\nSub acc: ', np.round(acc_all, 2))
    print('Avg acc: ', np.round(np.mean(acc_all), 2))

    acc_sub_str = str(np.round(acc_all, 2).tolist())
    acc_mean_str = str(np.round(np.mean(acc_all), 2).tolist())
    args.log.record("\n==========================================")
    args.log.record(acc_sub_str)
    args.log.record(acc_mean_str)

