# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import numpy as np
import argparse
import os
import torch as tr
import torch.nn as nn
import torch.optim as optim
from utils.LogRecord import LogRecord
from utils.dataloader import read_seed_src_tar
from utils.utils import fix_random_seed, data_load_noimg, op_copy, lr_scheduler
from utils import network, utils, loss


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_seed_src_tar(args)
    dset_loaders = data_load_noimg(X_src, y_src, X_tar, y_tar, args)

    netF_s, netC_s = network.backbone_net(args, args.bottleneck)
    netF_t, netC_t = network.backbone_net(args, args.bottleneck)

    mdl_type_list = ['netF', 'netC']
    if args.use_pretrain:
        mdl_path_name = [args.mdl_path + args.task_str + '/' + mdl + '.pt' for mdl in mdl_type_list]
    else:
        mdl_path_name = [args.mdl_path + mdl + '.pt' for mdl in mdl_type_list]
    netF_s.load_state_dict(tr.load(mdl_path_name[0]))
    netC_s.load_state_dict(tr.load(mdl_path_name[1]))

    netF_t.load_state_dict(tr.load(mdl_path_name[0]))
    netC_t.load_state_dict(tr.load(mdl_path_name[1]))

    param_group = []
    for k, v in netF_s.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
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
    for k, v in netC_t.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    sp_param_td = nn.Parameter(tr.tensor(5.0).cuda(), requires_grad=True)
    param_group += [{"params": [sp_param_td], "lr": args.lr}]
    optimizer_t = optim.SGD(param_group)
    optimizer_t = op_copy(optimizer_t)

    models_sd = nn.Sequential(netF_s, netC_s)
    models_td = nn.Sequential(netF_t, netC_t)

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

        pseudo_sd, top_prob_sd, threshold_sd = loss.get_target_preds(args, pred_tar_sd)
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
            netC_t.eval()

            acc_t_te, _ = utils.cal_acc_noimg(dset_loaders["Target"], netF_t, netC_t)
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, iter_num, max_iter, acc_t_te)
            print(log_str)

            netF_t.train()
            netC_t.train()

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

    # para for FixBi
    args.max_epoch = 50
    args.th = 2.0
    args.bim_start = 20
    args.sp_start = 20
    args.cr_start = 20
    args.lam_sd = 0.7
    args.lam_td = 0.3

    # para for train
    args.dset = data_name
    args.method = 'FixBi'
    args.backbone = 'ShallowNet'
    args.batch_size = 32  # 32
    args.input_dim = 310
    args.norm = 'zscore'
    args.noise_rate = 0
    dset_n = args.dset + '_' + str(args.noise_rate)
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'
    args.mdl_fixbi_dir = 'outputs/mdl_fixbi/' + dset_n + '/'
    args.use_pretrain = 1
    args.mdl_path = args.mdl_fixbi_dir if args.use_pretrain else args.mdl_init_dir

    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    args.data_env = 'gpu'  # 'local'
    args.seed = 2022
    fix_random_seed(args.seed)
    tr.backends.cudnn.deterministic = True

    print(dset_n, args.method)
    print(args)

    args.local_dir = r'/mnt/ssd2/wenz/NT-Benchmark/NT_UDA/'
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


