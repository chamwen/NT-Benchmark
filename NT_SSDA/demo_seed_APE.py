# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import numpy as np
import argparse
import os
import torch as tr
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import network, utils
from utils.LogRecord import LogRecord
from utils.dataloader import read_seed_src_tar
from utils.utils import lr_scheduler_full, fix_random_seed, data_load_noimg_ssda
from utils.loss import mix_rbf_mmd2, PerturbationGenerator_two, KLDivLossWithLogits
import warnings

warnings.filterwarnings('ignore')


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_seed_src_tar(args)
    dset_loaders = data_load_noimg_ssda(X_src, y_src, X_tar, y_tar, args)

    netF, netC = network.backbone_net(args, args.bottleneck)
    netF.load_state_dict(tr.load(args.mdl_init_dir + 'netF.pt'))
    netC.load_state_dict(tr.load(args.mdl_init_dir + 'netC.pt'))
    base_network = nn.Sequential(netF, netC)
    optimizer = optim.SGD(base_network.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders["source"])
    interval_iter = max_iter // 10
    args.max_iter = max_iter
    iter_num = 0

    netF.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source = iter_source.next()

        try:
            inputs_target_tr, labels_target_tr = iter_target_tr.next()
        except:
            iter_target_tr = iter(dset_loaders["target_tr"])
            inputs_target_tr, labels_target_tr = iter_target_tr.next()

        try:
            inputs_target, _ = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target_te"])
            inputs_target, _ = iter_target.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler_full(optimizer, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)

        criterion_supervision = nn.CrossEntropyLoss().cuda()
        criterion_reduce = nn.CrossEntropyLoss(reduce=False).cuda()
        criterion_consistency = KLDivLossWithLogits(reduction='mean').cuda()

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        inputs_target_tr, labels_target_tr = inputs_target_tr.cuda(), labels_target_tr.cuda()
        inputs_target = inputs_target.cuda()

        input_data_l = tr.cat((inputs_source, inputs_target_tr), 0)
        input_label_l = tr.cat((labels_source, labels_target_tr), 0)
        input_data_u = inputs_target

        latent_output_l = F.normalize(netF(input_data_l))
        latent_output_u = F.normalize(netF(input_data_u))
        _, output_l = netC(latent_output_l)
        _, output_u = netC(latent_output_u)
        args.temp = 0.05
        output_l, output_u = output_l / args.temp, output_u / args.temp

        # supervision loss:
        loss_supervision = criterion_supervision(output_l, input_label_l)

        # attraction loss:
        sigma = [1, 2, 5, 10]
        loss_attraction = 10 * mix_rbf_mmd2(latent_output_l, latent_output_u, sigma)

        # exploration loss:
        thr = 0.5
        pred = output_u.data.max(1)[1].detach()
        ent = - tr.sum(F.softmax(output_u, 1) * (tr.log(F.softmax(output_u, 1) + 1e-5)), 1)
        mask_reliable = (ent < thr).float().detach()
        loss_exploration = (mask_reliable * criterion_reduce(output_u, pred)).sum(0) / (1e-5 + mask_reliable.sum())

        # first optimization process:
        loss_1 = loss_supervision + loss_attraction + loss_exploration
        optimizer.zero_grad()
        loss_1.backward(retain_graph=False)
        optimizer.step()

        # perturbation loss:
        bs = labels_source.size(0)
        input_data_t = tr.cat((inputs_target_tr, inputs_target), 0)
        perturb, clean_vat_logits = PerturbationGenerator_two(netF, netC, xi=1, eps=25, ip=1)(input_data_t, args)
        perturb_inputs = input_data_t + perturb
        perturb_inputs = tr.cat(perturb_inputs.split(bs), 0)
        perturb_features = netF(perturb_inputs)
        perturb_logits = F.normalize(perturb_features)
        loss_perturbation = criterion_consistency(perturb_logits, clean_vat_logits)

        # second optimization process:
        optimizer.zero_grad()
        loss_2 = loss_perturbation * 10
        loss_2.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netC.eval()

            acc_t_te, _ = utils.cal_acc_noimg(dset_loaders["Target"], netF, netC)
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, iter_num, max_iter, acc_t_te)
            args.log.record(log_str)
            print(log_str)

            netF.train()
            netC.train()

    return acc_t_te


if __name__ == '__main__':

    data_name = 'SEED'
    if data_name == 'SEED': chn, class_num, trial_num = 62, 3, 3394
    focus_domain_idx = [0, 1, 2]
    domain_list = ['S' + str(i) for i in focus_domain_idx]
    num_domain = len(domain_list)

    args = argparse.Namespace(bottleneck=64, lr=0.01, epsilon=1e-05, layer='wn', smooth=0,
                              N=num_domain, chn=chn, class_num=class_num)

    args.dset = data_name
    args.method = 'APE'
    args.backbone = 'ShallowNet'
    args.batch_size = 32  # 32
    args.max_epoch = 50  # 50
    args.input_dim = 310
    args.norm = 'zscore'
    args.bz_tar_tr = args.batch_size
    args.bz_tar_te = args.batch_size * 2
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'
    args.noise_rate = 0
    dset_n = args.dset + '_' + str(args.noise_rate)
    args.tar_lbl_rate = 5  # [5, 10, ..., 50]/100

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    args.data_env = 'gpu'  # 'local'
    args.seed = 2022
    fix_random_seed(args.seed)
    tr.backends.cudnn.deterministic = True

    print(dset_n, args.method)
    print(args)

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
                print(args)

                my_log.record(info_str)
                args.log = my_log
                acc_all[itr_idx] = train_target(args)
    print('\nSub acc: ', np.round(acc_all, 3))
    print('Avg acc: ', np.round(np.mean(acc_all), 3))

    acc_sub_str = str(np.round(acc_all, 3).tolist())
    acc_mean_str = str(np.round(np.mean(acc_all), 3).tolist())
    args.log.record("\n==========================================")
    args.log.record(acc_sub_str)
    args.log.record(acc_mean_str)
