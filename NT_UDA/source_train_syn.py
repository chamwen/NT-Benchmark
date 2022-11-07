# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import numpy as np
import argparse
import os
import torch as tr
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import os.path as osp

from utils import network, loss, utils
from utils.LogRecord import LogRecord
from utils.dataloader import read_syn_single
from utils.utils import fix_random_seed, lr_scheduler_full, create_folder, add_label_noise_noimg


def data_load(X, y, args):
    dset_loaders = {}
    train_bs = args.batch_size

    if args.noise_rate > 0:
        y = add_label_noise_noimg(y, args.seed, args.class_num, args.noise_rate)

    args.validation = 'random'
    src_idx = np.arange(len(y.numpy()))
    if args.validation == 'random':
        num_train = int(0.9 * len(src_idx))
        tr.manual_seed(args.seed)
        id_train, id_val = tr.utils.data.random_split(src_idx, [num_train, len(src_idx) - num_train])

    source_tr = Data.TensorDataset(X[id_train, :], y[id_train])
    source_te = Data.TensorDataset(X[id_val, :], y[id_val])

    # for DAN/DANN/CDAN/MCC
    dset_loaders["source_tr"] = Data.DataLoader(source_tr, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["source_te"] = Data.DataLoader(source_te, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    return dset_loaders


def train_source(args):
    X_src, y_src = read_syn_single(args, args.src)
    dset_loaders = data_load(X_src, y_src, args)

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

        # same performance
        # classifier_loss = loss.CELabelSmooth(reduction='none', num_classes=class_num, epsilon=args.smooth)(
        #     outputs_source, labels_source)
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            acc_s_te = utils.cal_acc_base(dset_loaders["source_te"], base_network)
            log_str = 'Task: {}, Iter:{}/{}; Val_acc = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            print(log_str)

            base_network.train()

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netC = netC.state_dict()

    tr.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    tr.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return acc_s_te


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
                              ins_num=600)

    args.dset = data_name
    args.method = 'single'
    args.backbone = 'ShallowNet'
    args.batch_size = 32
    args.max_epoch = 50
    args.input_dim = 2
    args.eval_epoch = args.max_epoch / 10
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'
    args.noise_rate = 0
    dset_n = args.dset + '_' + str(args.noise_rate)

    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    args.data_env = 'gpu'  # 'local'
    args.seed = 2022
    fix_random_seed(args.seed)
    tr.backends.cudnn.deterministic = True

    mdl_path = 'outputs/models/'
    args.output = mdl_path + dset_n + '/source/'
    print(dset_n, args.method)
    print(args)

    args.root_path = './data_synth/'
    args.local_dir = r'/mnt/ssd2/wenz/code/NT-Benchmark/NT_UDA/'
    args.result_dir = 'results/source/'
    my_log = LogRecord(args)
    my_log.log_init()
    my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

    acc_all = []
    for s in range(num_domain):
        args.src = file_list[s]
        info_str = '\n========================== Within domain ' + domain_list[s] + ' =========================='
        print(info_str)
        my_log.record(info_str)
        args.log = my_log

        args.name_src = domain_list[s]
        args.output_dir_src = osp.join(args.output, args.name_src)
        create_folder(args.output_dir_src, args.data_env, args.local_dir)
        print(args)

        acc_sub = train_source(args)
        acc_all.append(acc_sub)
    print(np.round(acc_all, 2))
    print(np.round(np.mean(acc_all), 2))

    acc_sub_str = str(np.round(acc_all, 2).tolist())
    acc_mean_str = str(np.round(np.mean(acc_all), 2).tolist())
    args.log.record("\n==========================================")
    args.log.record(acc_sub_str)
    args.log.record(acc_mean_str)

