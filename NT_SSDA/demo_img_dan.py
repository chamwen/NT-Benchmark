# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import argparse
import os, sys
import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optim
import random, pdb, math, copy
import utils.network as network
from utils.loss import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel
from utils.utils import data_load_img_ssda, cal_acc_img, op_copy, lr_scheduler


def train_target(args):
    dset_loaders = data_load_img_ssda(args)
    # set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    netF.load_state_dict(tr.load(args.mdl_init_dir + 'netF.pt'))
    netB.load_state_dict(tr.load(args.mdl_init_dir + 'netB.pt'))
    netC.load_state_dict(tr.load(args.mdl_init_dir + 'netC.pt'))

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netC.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    netF.train()
    netB.train()
    netC.train()

    max_iter = args.max_epoch * len(dset_loaders["source"])
    interval_iter = max_iter // 10
    iter_num = 0

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
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        inputs_target = inputs_target.cuda()

        inputs_target_tr, labels_target_tr = inputs_target_tr.cuda(), labels_target_tr.cuda()
        _, outputs_source = netC(netB(netF(inputs_source)))
        _, outputs_target_tr = netC(netB(netF(inputs_target_tr)))
        outputs_comb = tr.cat((outputs_source, outputs_target_tr), dim=0)
        labels_comb = tr.cat((labels_source, labels_target_tr), dim=0)

        feas_source = netB(netF(inputs_source))
        feas_target_tr = netB(netF(inputs_target_tr))
        fea_comb = tr.cat((feas_source, feas_target_tr), dim=0)
        feas_target = netB(netF(inputs_target))

        # # loss definition
        classifier_loss = nn.CrossEntropyLoss()(outputs_comb, labels_comb)
        mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
            linear=not args.non_linear
        )
        discrepancy_loss = mkmmd_loss(fea_comb, feas_target)
        total_loss = classifier_loss + discrepancy_loss * args.trade_off

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            
            acc_t_te, _ = cal_acc_img(dset_loaders["Target"], netF, netB, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, iter_num, max_iter, acc_t_te)
            print(log_str)

            netF.train()
            netB.train()
            netC.train()

    return acc_t_te


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DAN')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=50, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VisDA17', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=1024)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--da', type=str, default='ssda', choices=['uda', 'pda'])
    parser.add_argument('--trte', type=str, default='val')
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--trade_off', type=float, default=1.0)
    parser.add_argument('--non_linear', default=False, action='store_true', help='whether not use the linear version')
    parser.add_argument('--total_repeat', type=int, default=1)
    args = parser.parse_args()

    args.dset = 'DomainNet'
    args.bz_tar_tr = args.batch_size
    args.bz_tar_te = args.batch_size * 2
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'
    args.noise_rate = 0
    dset_n = args.dset + '_' + str(args.noise_rate)
    args.tar_lbl_rate = 5  # [5, 10, ..., 50]/100

    if args.dset == 'DomainNet':
        domain_list = ['clipart', 'infograph', 'painting']
        args.class_num = 40  # 345

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # tr.backends.cudnn.deterministic = True
    args.data_env = 'gpu'  # 'local'

    for repeat in range(args.total_repeat):
        args.repeat = repeat
        args.seed = args.seed + repeat
        tr.manual_seed(args.seed)
        tr.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

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
                args.src, args.tar = s, t

                folder = "checkpoint/"
                args.s_dset_path = folder + args.dset + '/' + domain_list[args.src] + '_list.txt'
                args.t_dset_path = folder + args.dset + '/' + domain_list[args.tar] + '_list.txt'
                args.test_dset_path = folder + args.dset + '/' + domain_list[args.tar] + '_list.txt'
                args.task_str = domain_list[args.src][0].upper() + domain_list[args.tar][0].upper()
                print(args)

                acc_all[itr_idx] = train_target(args)
                print('done\n')
        print('\n\nfinish one repeat')
        print('All acc: ', np.round(acc_all, 2))
        print('Avg acc: ', np.round(np.mean(acc_all), 2))

