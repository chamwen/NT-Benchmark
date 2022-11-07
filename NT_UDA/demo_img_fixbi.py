# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import argparse
import os
import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optim
import random
from utils.utils import data_load_img, cal_acc_img, op_copy, lr_scheduler
from utils import network, utils, loss


def train_target(args):
    dset_loaders = data_load_img(args)

    # set base network
    if args.net[0:3] == 'res':
        netF_s = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF_s = network.VGGBase(vgg_name=args.net).cuda()

    netB_s = network.feat_bottleneck(type=args.classifier, feature_dim=netF_s.in_features,
                                     bottleneck_dim=args.bottleneck).cuda()
    netC_s = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    if args.net[0:3] == 'res':
        netF_t = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF_t = network.VGGBase(vgg_name=args.net).cuda()

    netB_t = network.feat_bottleneck(type=args.classifier, feature_dim=netF_t.in_features,
                                     bottleneck_dim=args.bottleneck).cuda()
    netC_t = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    mdl_type_list = ['netF', 'netB', 'netC']
    if args.use_pretrain:
        mdl_path_name = [args.mdl_path + args.task_str + '/' + mdl + '_' + str(args.repeat) + '.pt' for mdl in
                         mdl_type_list]
    else:
        mdl_path_name = [args.mdl_path + mdl + '.pt' for mdl in mdl_type_list]
    netF_s.load_state_dict(tr.load(mdl_path_name[0]))
    netB_s.load_state_dict(tr.load(mdl_path_name[1]))
    netC_s.load_state_dict(tr.load(mdl_path_name[2]))

    netF_t.load_state_dict(tr.load(mdl_path_name[0]))
    netB_t.load_state_dict(tr.load(mdl_path_name[1]))
    netC_t.load_state_dict(tr.load(mdl_path_name[2]))

    param_group = []
    for k, v in netF_s.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB_s.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netC_s.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    sp_param_sd = nn.Parameter(tr.tensor(5.0).cuda(), requires_grad=True)
    param_group += [{"params": [sp_param_sd], "lr": args.lr}]
    optimizer_s = optim.SGD(param_group)
    optimizer_s = op_copy(optimizer_s)

    param_group = []
    for k, v in netF_t.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB_t.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netC_t.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    sp_param_td = nn.Parameter(tr.tensor(5.0).cuda(), requires_grad=True)
    param_group += [{"params": [sp_param_td], "lr": args.lr}]
    optimizer_t = optim.SGD(param_group)
    optimizer_t = op_copy(optimizer_t)

    models_sd = nn.Sequential(netF_s, netB_s, netC_s)
    models_td = nn.Sequential(netF_t, netB_t, netC_t)

    ce = nn.CrossEntropyLoss().cuda()
    mse = nn.MSELoss().cuda()

    models_sd.train()
    models_td.train()

    num_batch = len(dset_loaders["source"])
    max_iter = args.max_epoch * num_batch
    interval_iter = max_iter // 10
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source = iter_source.next()

        try:
            inputs_target, labels_target = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_target, labels_target = iter_target.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer_s, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_t, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        inputs_target = inputs_target.cuda()

        inputs_source, labels_source = inputs_source.cuda(non_blocking=True), labels_source.cuda(non_blocking=True)
        inputs_target, labels_target = inputs_target.cuda(non_blocking=True), labels_target.cuda(non_blocking=True)
        
        _, pred_tar_sd = models_sd(inputs_target)
        _, pred_tar_td = models_td(inputs_target)

        pseudo_sd, top_prob_sd, threshold_sd = loss.get_target_preds(args, pred_tar_sd)  # for target
        fixmix_sd_loss = loss.get_fixmix_loss(models_sd, inputs_source, inputs_target, labels_source, pseudo_sd,
                                              args.lam_sd)

        pseudo_td, top_prob_td, threshold_td = loss.get_target_preds(args, pred_tar_td)
        fixmix_td_loss = loss.get_fixmix_loss(models_td, inputs_source, inputs_target, labels_source, pseudo_td,
                                              args.lam_td)

        total_loss = fixmix_sd_loss + fixmix_td_loss

        if iter_num == 0:
            print('Fixed-mixup Loss, sdm: {:.4f}, tdm: {:.4f}'.format(fixmix_sd_loss.item(), fixmix_td_loss.item()))

        # Bidirectional Matching
        if iter_num // num_batch > args.bim_start:
            bim_mask_sd = tr.ge(top_prob_sd, threshold_sd)
            bim_mask_sd = tr.nonzero(bim_mask_sd).squeeze()

            bim_mask_td = tr.ge(top_prob_td, threshold_td)
            bim_mask_td = tr.nonzero(bim_mask_td).squeeze()

            if bim_mask_sd.dim() > 0 and bim_mask_td.dim() > 0:
                if bim_mask_sd.numel() > 0 and bim_mask_td.numel() > 0:
                    bim_mask = min(bim_mask_sd.size(0), bim_mask_td.size(0))
                    bim_sd_loss = ce(pred_tar_sd[bim_mask_td[:bim_mask]], pseudo_td[bim_mask_td[:bim_mask]].cuda().detach())
                    bim_td_loss = ce(pred_tar_td[bim_mask_sd[:bim_mask]], pseudo_sd[bim_mask_sd[:bim_mask]].cuda().detach())

                    total_loss += bim_sd_loss
                    total_loss += bim_td_loss

                    if iter_num == 0:
                        print('Bidirectional Loss sdm: {:.4f}, tdm: {:.4f}'.format(bim_sd_loss.item(),
                                                                                   bim_td_loss.item()))

        # Self-penalization
        if iter_num // num_batch <= args.sp_start:
            sp_mask_sd = tr.lt(top_prob_sd, threshold_sd)
            sp_mask_sd = tr.nonzero(sp_mask_sd).squeeze()

            sp_mask_td = tr.lt(top_prob_sd, threshold_td)
            sp_mask_td = tr.nonzero(sp_mask_td).squeeze()

            if sp_mask_sd.dim() > 0 and sp_mask_td.dim() > 0:
                if sp_mask_sd.numel() > 0 and sp_mask_td.numel() > 0:
                    sp_mask = min(sp_mask_sd.size(0), sp_mask_td.size(0))
                    sp_sd_loss = loss.get_sp_loss(pred_tar_sd[sp_mask_sd[:sp_mask]], pseudo_sd[sp_mask_sd[:sp_mask]],
                                                  sp_param_sd)
                    sp_td_loss = loss.get_sp_loss(pred_tar_td[sp_mask_td[:sp_mask]], pseudo_td[sp_mask_td[:sp_mask]],
                                                  sp_param_td)

                    total_loss += sp_sd_loss
                    total_loss += sp_td_loss

                    if iter_num == 0:
                        print('Penalization Loss sdm: {:.4f}, tdm: {:.4f}', sp_sd_loss.item(), sp_td_loss.item())

        # Consistency Regularization
        if iter_num // num_batch > args.cr_start:
            mixed_cr = 0.5 * inputs_source + 0.5 * inputs_target
            _, out_sd = models_sd(mixed_cr)
            _, out_td = models_td(mixed_cr)
            cr_loss = mse(out_sd, out_td)
            total_loss += cr_loss
            if iter_num == 0:
                print('Consistency Loss: {:.4f}', cr_loss.item())

        optimizer_s.zero_grad()
        optimizer_t.zero_grad()
        total_loss.backward()
        optimizer_s.step()
        optimizer_t.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF_t.eval()
            netB_t.eval()
            netC_t.eval()

            acc_t_te, _ = cal_acc_img(dset_loaders["Target"], netF_t, netB_t, netC_t, False)
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, iter_num, max_iter, acc_t_te)
            print(log_str)

            netF_t.train()
            netB_t.train()
            netC_t.train()

    return acc_t_te


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FixBi')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
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
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--trte', type=str, default='val')
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--trade_off', type=float, default=1.0)
    parser.add_argument('--total_repeat', type=int, default=1)
    args = parser.parse_args()

    # para for FixBi
    args.max_epoch = 50
    args.th = 2.0
    args.bim_start = 20
    args.sp_start = 20
    args.cr_start = 20
    args.lam_sd = 0.7
    args.lam_td = 0.3

    args.dset = 'DomainNet'
    args.noise_rate = 0
    dset_n = args.dset + '_' + str(args.noise_rate)
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'
    args.mdl_fixbi_dir = 'outputs/mdl_fixbi/' + dset_n + '/'
    args.use_pretrain = 1
    args.mdl_path = args.mdl_fixbi_dir if args.use_pretrain else args.mdl_init_dir

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

