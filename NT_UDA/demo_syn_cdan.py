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
from utils.network import calc_coeff
from utils.dataloader import read_syn_src_tar
from utils.utils import lr_scheduler_full, fix_random_seed, data_load_noimg
from utils.loss import CELabelSmooth, CDANE, Entropy, RandomLayer


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_syn_src_tar(args)
    dset_loaders = data_load_noimg(X_src, y_src, X_tar, y_tar, args)

    netF, netC = network.backbone_net(args, args.bottleneck)
    netF.load_state_dict(tr.load(args.mdl_init_dir + 'netF.pt'))
    netC.load_state_dict(tr.load(args.mdl_init_dir + 'netC.pt'))
    base_network = nn.Sequential(netF, netC)

    args.max_iter = len(dset_loaders["source"])

    ad_net = network.AdversarialNetwork(args.bottleneck, 20).cuda()
    ad_net.load_state_dict(tr.load(args.mdl_init_dir + 'netD_full.pt'))
    random_layer = RandomLayer([args.bottleneck, args.class_num], args.bottleneck)
    random_layer.cuda()

    optimizer_f = optim.SGD(netF.parameters(), lr=args.lr * 0.1)
    optimizer_c = optim.SGD(netC.parameters(), lr=args.lr)
    optimizer_d = optim.SGD(ad_net.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders["source"])
    interval_iter = max_iter // 10
    args.max_iter = args.max_epoch * len(dset_loaders["source"])
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
        lr_scheduler_full(optimizer_f, init_lr=args.lr * 0.1, iter_num=iter_num, max_iter=args.max_iter)
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
        total_loss = args.loss_trade_off * transfer_loss + classifier_loss

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

    data_name = 'moon'
    if data_name == 'moon': num_class = 2
    base_name_list = ['0', '1', '2', '3_45', '4_15', '6', '7', '8', '9']
    domain_list = ['Raw', 'Tl', 'Sl', 'Rt', 'Sh', 'Sk', 'Ns', 'Ol', 'Sc']
    file_list = [data_name + i for i in base_name_list]
    num_domain = len(domain_list)

    args = argparse.Namespace(bottleneck=64, lr=0.01, lr_decay1=0.1, lr_decay2=1.0,
                              epsilon=1e-05, layer='wn', class_num=num_class, smooth=0)

    args.method = 'CDAN'
    args.dset = data_name
    args.backbone = 'ShallowNet'
    args.batch_size = 32
    args.max_epoch = 50
    args.input_dim = 2
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'
    args.noise_rate = 0
    dset_n = args.dset + '_' + str(args.noise_rate)

    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    args.data_env = 'gpu'  # 'local'
    args.seed = 2022
    fix_random_seed(args.seed)
    tr.backends.cudnn.deterministic = True
    print(dset_n, args.method)

    args.root_path = './data_synth/'
    args.local_dir = r'/mnt/ssd2/wenz/NT-Benchmark/NT_UDA/'
    args.result_dir = 'results/target/'

    acc_all = np.zeros((len(domain_list) - 1))
    for s in range(1, num_domain):  # source
        for t in [0]:  # target
            itr_idx = s - 1
            info_str = '\n%s: %s --> %s' % (itr_idx, domain_list[s], domain_list[t])
            print(info_str)
            args.src, args.tar = file_list[s], file_list[t]
            args.task_str = domain_list[s] + '_' + domain_list[t]
            print(args)

            acc_all[itr_idx] = train_target(args)
    print('All acc: ', np.round(acc_all, 2))
    print('Avg acc: ', np.round(np.mean(acc_all), 2))

