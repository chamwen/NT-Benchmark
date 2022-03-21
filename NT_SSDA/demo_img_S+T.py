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
import random, pdb, math, copy
import utils.network as network
from utils.utils import data_load_img_ssda, save_fea_base_ssda, op_copy, lr_scheduler, cal_acc_img


def train_source_test_target(args):
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

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
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

        feas = netB(netF(inputs_data))
        _, output = netC(feas)
        classifier_loss = nn.CrossEntropyLoss()(output, inputs_label)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % (interval_iter * 2) == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()

            acc_s_te, _ = cal_acc_img(dset_loaders['source_te'], netF, netB, netC, False)
            acc_t_te, _ = cal_acc_img(dset_loaders['Target'], netF, netB, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Val_Acc = {:.2f}%; Test_Acc = {:.2f}%'.format(
                args.task_str, iter_num, max_iter, acc_s_te, acc_t_te)
            print(log_str)

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                acc_tar_src_best = acc_t_te
                best_netF = netF.cpu().state_dict()
                best_netB = netB.cpu().state_dict()
                best_netC = netC.cpu().state_dict()
                netF.cuda()
                netB.cuda()
                netC.cuda()

            netF.train()
            netB.train()
            netC.train()

    tr.save(best_netF, osp.join(args.output_dir, "sourceF" + "_" + str(args.repeat) + ".pt"))
    tr.save(best_netB, osp.join(args.output_dir, "sourceB" + "_" + str(args.repeat) + ".pt"))
    tr.save(best_netC, osp.join(args.output_dir, "sourceC" + "_" + str(args.repeat) + ".pt"))

    return acc_tar_src_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ST')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
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
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--trte', type=str, default='val')
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--total_repeat', type=int, default=1)
    args = parser.parse_args()

    args.dset = 'DomainNet'
    args.bz_tar_tr = int(args.batch_size / 2)
    args.bz_tar_te = args.batch_size
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'
    args.noise_rate = 0
    dset_n = args.dset + '_' + str(args.noise_rate)
    args.tar_lbl_rate = 5  # [5, 10, ..., 50]/100

    if args.dset == 'DomainNet':
        domain_list = ['clipart', 'infograph', 'painting']
        args.class_num = 40  # 345

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    tr.backends.cudnn.deterministic = True
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

                mdl_dir = 'outputs/mdl_fea'
                args.output_dir = osp.join(mdl_dir, dset_n, args.task_str + '_tr' + str(args.tar_lbl_rate))
                if not osp.exists(args.output_dir):
                    os.system('mkdir -p ' + args.output_dir)
                if not osp.exists(args.output_dir):
                    os.mkdir(args.output_dir)

                acc_all[itr_idx] = train_source_test_target(args)

                args.fea_dir = 'outputs/feas/' + dset_n + '/' + 'tr' + str(args.tar_lbl_rate) + '/'
                if not osp.exists(args.fea_dir):
                    os.system('mkdir -p ' + args.fea_dir)
                if not osp.exists(args.fea_dir):
                    os.mkdir(args.fea_dir)
                save_fea_base_ssda(args)

                print('done\n')
        print('\n\nfinish one repeat')
        print('All acc: ', np.round(acc_all, 2))
        print('Avg acc: ', np.round(np.mean(acc_all), 2))

