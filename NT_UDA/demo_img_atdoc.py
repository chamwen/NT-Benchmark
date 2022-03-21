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
from torch.utils.data import DataLoader
from utils.network import AdversarialNetwork, calc_coeff
from utils.loss import CDANE, CELabelSmooth, Entropy, RandomLayer
from utils.utils import cal_acc_img, op_copy, lr_scheduler
from utils.utils import image_train, image_test, add_label_noise_img
from utils.data_list import ImageList_idx, ImageList


def data_load(args):
    # prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()

    # only add for source domain
    if args.noise_rate > 0:
        txt_src = add_label_noise_img(args, txt_src)

    dsets["source"] = ImageList(txt_src, transform=image_train())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=True)
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True)
    dsets["Target"] = ImageList(txt_tar, transform=image_test())
    dset_loaders["Target"] = DataLoader(dsets["Target"], batch_size=train_bs * 3, shuffle=False,
                                        num_workers=args.worker, drop_last=False)

    return dset_loaders


def train_target(args):
    dset_loaders = data_load(args)
    # set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netD = AdversarialNetwork(args.bottleneck, 2048).cuda()

    max_len = max(len(dset_loaders["source"]), len(dset_loaders["target"]))
    args.max_iter = args.max_epoch * max_len

    netF.load_state_dict(tr.load(args.mdl_init_dir + 'netF.pt'))
    netB.load_state_dict(tr.load(args.mdl_init_dir + 'netB.pt'))
    netC.load_state_dict(tr.load(args.mdl_init_dir + 'netC.pt'))
    netD.load_state_dict(tr.load(args.mdl_init_dir + 'netD_full.pt'))

    random_layer = RandomLayer([args.bottleneck, args.class_num], args.bottleneck)
    random_layer.cuda()

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
    for k, v in netD.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    netF.train()
    netB.train()
    netC.train()
    netD.train()

    max_len = max(len(dset_loaders["source"]), len(dset_loaders["target"]))
    max_iter = args.max_epoch * max_len
    interval_iter = max_iter // 10
    iter_num = 0

    class_num = args.class_num
    mem_fea = tr.rand(len(dset_loaders["target"].dataset), args.bottleneck).cuda()
    mem_fea = mem_fea / tr.norm(mem_fea, p=2, dim=1, keepdim=True)
    mem_cls = tr.ones(len(dset_loaders["target"].dataset), class_num).cuda() / class_num

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source = iter_source.next()

        try:
            inputs_target, _, idx = target_loader_iter.next()
        except:
            target_loader_iter = iter(dset_loaders["target"])
            inputs_target, _, idx = target_loader_iter.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        inputs_target = inputs_target.cuda()
        feas_source, feas_target = netB(netF(inputs_source)), netB(netF(inputs_target))

        _, outputs_source = netC(feas_source)
        _, outputs_target = netC(feas_target)
        features = tr.cat((feas_source, feas_target), dim=0)
        outputs = tr.cat((outputs_source, outputs_target), dim=0)

        # # loss definition
        # p = float(iter_num) / max_iter
        # alpha = 2. / (1. + np.exp(-10 * p)) - 1
        softmax_out = nn.Softmax(dim=1)(outputs)
        entropy = Entropy(softmax_out)
        transfer_loss = CDANE([features, softmax_out], netD, entropy, calc_coeff(iter_num), random_layer=random_layer)
        classifier_loss = CELabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)

        # ATDOC
        dis = -tr.mm(feas_target.detach(), mem_fea.t())
        for di in range(dis.size(0)):
            dis[di, idx[di]] = tr.max(dis)
        _, p1 = tr.sort(dis, dim=1)

        w = tr.zeros(feas_target.size(0), mem_fea.size(0)).cuda()
        for wi in range(w.size(0)):
            for wj in range(args.K):
                w[wi][p1[wi, wj]] = 1 / args.K

        weight_, pred = tr.max(w.mm(mem_cls), 1)
        loss_ = nn.CrossEntropyLoss(reduction='none')(outputs_target, pred)
        classifier_loss_atdoc = tr.sum(weight_ * loss_) / (tr.sum(weight_).item())

        eff = iter_num / args.max_iter
        total_loss = args.loss_trade_off * transfer_loss + classifier_loss + args.tar_par * eff * classifier_loss_atdoc

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # label memory
        netF.eval()
        netB.eval()
        netC.eval()
        with tr.no_grad():
            features_target, outputs_target = netC(netB(netF(inputs_target)))
            features_target = features_target / tr.norm(features_target, p=2, dim=1, keepdim=True)
            softmax_out = nn.Softmax(dim=1)(outputs_target)
            outputs_target = softmax_out ** 2 / ((softmax_out ** 2).sum(dim=0))

        mem_fea[idx] = (1.0 - args.momentum) * mem_fea[idx] + args.momentum * features_target.clone()
        mem_cls[idx] = (1.0 - args.momentum) * mem_cls[idx] + args.momentum * outputs_target.clone()

        if iter_num % (interval_iter * 2) == 0 or iter_num == max_iter:
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
    parser = argparse.ArgumentParser(description='CDANE-ATDOC')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='5', help="device id to run")
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
    parser.add_argument('--lr_decay2', type=float, default=0.1)

    parser.add_argument('--bottleneck', type=int, default=1024)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--trte', type=str, default='val')
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--loss_trade_off', type=float, default=1.0)
    parser.add_argument('--total_repeat', type=int, default=1)
    args = parser.parse_args()

    args.K = 5
    args.momentum = 1.0
    args.tar_par = 0.2

    args.dset = 'DomainNet'
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'
    args.noise_rate = 0
    dset_n = args.dset + '_' + str(args.noise_rate)
    if args.dset == 'DomainNet':
        domain_list = ['clipart', 'infograph', 'painting']
        args.class_num = 40  # 345

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
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
