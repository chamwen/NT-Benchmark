# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import numpy as np
import argparse
import os
import torch as tr
import torch.nn as nn
import torch.optim as optim
from utils import network, loss, utils
from utils.dataloader import read_seed_src_tar
from utils.utils import lr_scheduler_full, fix_random_seed, data_load_noimg
from utils.loss import CELabelSmooth, ReverseLayerF


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_seed_src_tar(args)
    dset_loaders = data_load_noimg(X_src, y_src, X_tar, y_tar, args)

    netF, netC = network.backbone_net(args, args.bottleneck)
    netF.load_state_dict(tr.load(args.mdl_init_dir + 'netF.pt'))
    netC.load_state_dict(tr.load(args.mdl_init_dir + 'netC.pt'))
    base_network = nn.Sequential(netF, netC)

    args.max_iter = args.max_epoch * len(dset_loaders["source"])

    ad_net = network.feat_classifier(type=args.layer, class_num=2, bottleneck_dim=args.bottleneck).cuda()
    ad_net.load_state_dict(tr.load(args.mdl_init_dir + 'netD_clf.pt'))

    optimizer_f = optim.SGD(netF.parameters(), lr=args.lr)
    optimizer_c = optim.SGD(netC.parameters(), lr=args.lr)
    optimizer_d = optim.SGD(ad_net.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders["source"])
    interval_iter = max_iter // 10
    args.max_iter = max_iter
    iter_num = 0
    base_network.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source = iter_source.next()

        try:
            inputs_target, _ = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_target, _ = iter_target.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler_full(optimizer_f, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler_full(optimizer_c, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler_full(optimizer_d, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)

        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)

        # new version img loss
        p = float(iter_num) / max_iter
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        reverse_source, reverse_target = ReverseLayerF.apply(features_source, alpha), ReverseLayerF.apply(
            features_target,
            alpha)
        _, domain_output_s = ad_net(reverse_source)
        _, domain_output_t = ad_net(reverse_target)
        domain_label_s = tr.ones(inputs_source.size()[0]).long().cuda()
        domain_label_t = tr.zeros(inputs_target.size()[0]).long().cuda()

        classifier_loss = CELabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)
        adv_loss = nn.CrossEntropyLoss()(domain_output_s, domain_label_s) + nn.CrossEntropyLoss()(domain_output_t,
                                                                                                  domain_label_t)
        total_loss = classifier_loss + adv_loss

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        optimizer_d.zero_grad()
        total_loss.backward()
        optimizer_f.step()
        optimizer_c.step()
        optimizer_d.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            acc_t_te = utils.cal_acc_base(dset_loaders["Target"], base_network)
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, iter_num, max_iter, acc_t_te)
            print(log_str)

            base_network.train()

    return acc_t_te


if __name__ == '__main__':

    data_name = 'SEED'
    if data_name == 'SEED': chn, class_num, trial_num = 62, 3, 3394
    focus_domain_idx = [0, 1, 2]
    domain_list = ['S' + str(i) for i in focus_domain_idx]
    num_domain = len(domain_list)

    args = argparse.Namespace(bottleneck=64, lr=0.01, lr_decay1=0.1, lr_decay2=1.0,
                              epsilon=1e-05, layer='wn', smooth=0,
                              N=num_domain, chn=chn, class_num=class_num)

    args.dset = data_name
    args.method = 'DANN'
    args.backbone = 'ShallowNet'
    args.batch_size = 32  # 32
    args.max_epoch = 50  # 50 enough to converge
    args.input_dim = 310
    args.norm = 'zscore'
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    args.data_env = 'gpu'  # 'local'
    args.seed = 2022  # 2021~2023 repeat three times
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

        acc_all[ns] = train_target(args)
    print('\nSub acc: ', np.round(acc_all, 3))
    print('Avg acc: ', np.round(np.mean(acc_all), 3))

    acc_sub_str = str(np.round(acc_all, 3).tolist())
    acc_mean_str = str(np.round(np.mean(acc_all), 3).tolist())

