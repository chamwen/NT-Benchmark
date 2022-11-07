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
from utils.LogRecord import LogRecord
from utils.dataloader import read_seed_src_tar
from utils.utils import lr_scheduler_full, fix_random_seed, add_label_noise_noimg
from utils.loss import CELabelSmooth, CDANE, Entropy, RandomLayer
from utils.network import calc_coeff
import torch.utils.data as Data


def data_load(Xs, Ys, Xt, Yt, args):
    dset_loaders = {}
    train_bs = args.batch_size

    if args.noise_rate > 0:
        Ys = add_label_noise_noimg(Ys, args.seed, args.class_num, args.noise_rate)

    sample_idx_tar = tr.from_numpy(np.arange(len(Yt))).long()
    data_src = Data.TensorDataset(Xs, Ys)
    data_tar = Data.TensorDataset(Xt, Yt)
    data_tar_idx = Data.TensorDataset(Xt, Yt, sample_idx_tar)

    # for DAN/DANN/CDAN/MCC
    dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["target"] = Data.DataLoader(data_tar_idx, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    return dset_loaders


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_seed_src_tar(args)
    dset_loaders = data_load(X_src, y_src, X_tar, y_tar, args)

    netF, netC = network.backbone_net(args, args.bottleneck)
    netF.load_state_dict(tr.load(args.mdl_init_dir + 'netF.pt'))
    netC.load_state_dict(tr.load(args.mdl_init_dir + 'netC.pt'))
    base_network = nn.Sequential(netF, netC)

    max_len = max(len(dset_loaders["source"]), len(dset_loaders["target"]))
    args.max_iter = args.max_epoch * max_len

    ad_net = network.AdversarialNetwork(args.bottleneck, 20).cuda()
    ad_net.load_state_dict(tr.load(args.mdl_init_dir + 'netD_full.pt'))
    random_layer = RandomLayer([args.bottleneck, args.class_num], args.bottleneck)
    random_layer.cuda()

    optimizer_f = optim.SGD(netF.parameters(), lr=args.lr)
    optimizer_c = optim.SGD(netC.parameters(), lr=args.lr)
    optimizer_d = optim.SGD(ad_net.parameters(), lr=args.lr)

    max_len = max(len(dset_loaders["source"]), len(dset_loaders["target"]))
    max_iter = args.max_epoch * max_len
    interval_iter = max_iter // 10
    iter_num = 0
    base_network.train()

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
            inputs_target, _, idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_target, _, idx = iter_target.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler_full(optimizer_f, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler_full(optimizer_c, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler_full(optimizer_d, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)

        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = tr.cat((features_source, features_target), dim=0)

        # new version img loss
        args.loss_trade_off = 1.0
        outputs = tr.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        entropy = Entropy(softmax_out)
        transfer_loss = CDANE([features, softmax_out], ad_net, entropy, calc_coeff(iter_num), random_layer=random_layer)
        classifier_loss = CELabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)

        # ATDOC
        dis = -tr.mm(features_target.detach(), mem_fea.t())
        for di in range(dis.size(0)):
            dis[di, idx[di]] = tr.max(dis)
        _, p1 = tr.sort(dis, dim=1)

        w = tr.zeros(features_target.size(0), mem_fea.size(0)).cuda()
        for wi in range(w.size(0)):
            for wj in range(args.K):
                w[wi][p1[wi, wj]] = 1 / args.K

        weight_, pred = tr.max(w.mm(mem_cls), 1)
        loss_ = nn.CrossEntropyLoss(reduction='none')(outputs_target, pred)
        classifier_loss_atdoc = tr.sum(weight_ * loss_) / (tr.sum(weight_).item())

        eff = iter_num / args.max_iter
        total_loss = args.loss_trade_off * transfer_loss + classifier_loss + args.tar_par * eff * classifier_loss_atdoc

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        optimizer_d.zero_grad()
        total_loss.backward()
        optimizer_f.step()
        optimizer_c.step()
        optimizer_d.step()

        # label memory
        netF.eval()
        netC.eval()
        with tr.no_grad():
            features_target, outputs_target = netC(netF(inputs_target))
            features_target = features_target / tr.norm(features_target, p=2, dim=1, keepdim=True)
            softmax_out = nn.Softmax(dim=1)(outputs_target)
            outputs_target = softmax_out ** 2 / ((softmax_out ** 2).sum(dim=0))

        mem_fea[idx] = (1.0 - args.momentum) * mem_fea[idx] + args.momentum * features_target.clone()
        mem_cls[idx] = (1.0 - args.momentum) * mem_cls[idx] + args.momentum * outputs_target.clone()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            acc_t_te = utils.cal_acc_base(dset_loaders["Target"], base_network)
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, iter_num, max_iter, acc_t_te)
            args.log.record(log_str)
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

    args.K = 5
    args.momentum = 1.0
    args.tar_par = 0.2

    args.dset = data_name
    args.method = 'CDANE-ATDOC'
    args.backbone = 'ShallowNet'
    args.batch_size = 32  # 32
    args.max_epoch = 50  # 50
    args.input_dim = 310
    args.norm = 'zscore'
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'
    args.eval_epoch = args.max_epoch / 10
    args.noise_rate = 0
    dset_n = args.dset + '_' + str(args.noise_rate)

    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
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

