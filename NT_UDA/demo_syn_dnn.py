# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import numpy as np
import argparse
import random
import os
import torch as tr
import torch.nn as nn
import torch.optim as optim
from utils import network, loss, utils
from utils.dataloader import read_syn_src_tar
from utils.utils import fix_random_seed, lr_scheduler_full, data_load_noimg


def train_source_test_target(args):
    X_src, y_src, X_tar, y_tar = read_syn_src_tar(args)
    dset_loaders = data_load_noimg(X_src, y_src, X_tar, y_tar, args)

    netF, netC = network.backbone_net(args, args.bottleneck)
    netF.load_state_dict(tr.load(args.mdl_init_dir + 'netF.pt'))
    netC.load_state_dict(tr.load(args.mdl_init_dir + 'netC.pt'))
    base_network = nn.Sequential(netF, netC)
    optimizer = optim.SGD(base_network.parameters(), lr=args.lr)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    args.max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    iter_num = 0
    base_network.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = source_loader_iter.next()
        except:
            source_loader_iter = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = source_loader_iter.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler_full(optimizer, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)

        # classifier_loss = loss.CELabelSmooth(reduction='none', num_classes=class_num, epsilon=args.smooth)(
        #     outputs_source, labels_source)
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            acc_s_te = utils.cal_acc_base(dset_loaders["source_te"], base_network)
            acc_t_te = utils.cal_acc_base(dset_loaders["Target"], base_network)
            log_str = 'Task: {}, Iter:{}/{}; Val_acc = {:.2f}%; Test_Acc = {:.2f}%'.format(args.task_str, iter_num,
                                                                                           max_iter, acc_s_te, acc_t_te)
            print(log_str)
            base_network.train()

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                acc_tar_src_best = acc_t_te

    return acc_tar_src_best


if __name__ == '__main__':

    data_name = 'moon'
    seed = 2022
    if data_name == 'moon': num_class = 2
    noise_rate = 0
    base_name_list = ['0', '1', '2', '3_45', '4_15', '6', '7', '8', '9']
    domain_list = ['Raw', 'Tl', 'Sl', 'Rt', 'Sh', 'Sk', 'Ns', 'Ol', 'Sc']
    file_list = [data_name + i for i in base_name_list]
    num_domain = len(domain_list)

    args = argparse.Namespace(bottleneck=64, lr=0.01, lr_decay1=0.1, lr_decay2=1.0,
                              epsilon=1e-05, layer='wn', class_num=num_class, smooth=0,
                              is_save=False, ins_num=600)

    args.method = 'DNN'
    args.dset = data_name
    args.backbone = 'ShallowNet'
    args.batch_size = 32
    args.max_epoch = 50
    args.input_dim = 2
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'
    args.noise_rate = 0
    dset_n = args.dset + '_' + str(args.noise_rate)

    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    args.data_env = 'gpu'  # 'local'
    args.seed = 2022
    fix_random_seed(args.seed)
    tr.backends.cudnn.deterministic = True
    print(dset_n, args.method)

    args.root_path = './data_synth/'
    args.local_dir = r'/mnt/ssd2/wenz/NT-Benchmark/NT_UDA/'
    args.result_dir = 'results/target/'

    tr.manual_seed(args.seed)
    tr.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    acc_all = np.zeros((len(domain_list) - 1))
    for s in range(1, num_domain):  # source
        for t in [0]:  # target
            itr_idx = s - 1
            info_str = '\n%s: %s --> %s' % (itr_idx, domain_list[s], domain_list[t])
            print(info_str)
            args.src, args.tar = file_list[s], file_list[t]
            args.task_str = domain_list[s] + '_' + domain_list[t]
            print(args)

            acc_all[itr_idx] = train_source_test_target(args)
            print('done\n')
    print('\n\nfinish one repeat')
    print('\nAll acc: ', np.round(acc_all, 2))
    print('Avg acc: ', np.round(np.mean(acc_all), 2))
