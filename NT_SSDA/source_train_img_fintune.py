# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import argparse
import os, sys
import os.path as osp
import numpy as np
import torch as tr
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.network as network
from utils.data_list import ImageList
from utils.LogRecord import LogRecord
from utils.loss import CELabelSmooth, Entropy
from utils.utils import cal_acc_img, op_copy, lr_scheduler, add_label_noise_img, \
    image_train, image_test, fix_random_seed, create_folder


def data_load(args):
    # prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()

    if args.noise_rate > 0:
        txt_src = add_label_noise_img(args, txt_src)

    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        tr.manual_seed(args.seed)
        src_tr_txt, src_te_txt = tr.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        tr.manual_seed(args.seed)
        _, src_te_txt = tr.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        src_tr_txt = txt_src

    # for DNN
    dsets["source_tr"] = ImageList(src_tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=True)
    dsets["source_te"] = ImageList(src_te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=False,
                                           num_workers=args.worker, drop_last=False)

    return dset_loaders


def train_source(args):
    dset_loaders = data_load(args)
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

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        feas_source = netB(netF(inputs_source))
        _, outputs_source = netC(feas_source)
        classifier_loss = CELabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source,
                                                                                         labels_source)
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()

            acc_s_te, _ = cal_acc_img(dset_loaders['source_te'], netF, netB, netC)
            log_str = 'Task: {}, Iter:{}/{}; Val_acc = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.log.record(log_str)
            print(log_str)

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netB.train()
            netC.train()

    tr.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    tr.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
    tr.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return acc_s_te


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre-train')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='5', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")  # S-->Q
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
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--trte', type=str, default='val')
    parser.add_argument('--smooth', type=float, default=0.1)
    args = parser.parse_args()

    args.dset = 'DomainNet'
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'
    args.noise_rate = 0
    dset_n = args.dset + '_' + str(args.noise_rate)

    if args.dset == 'DomainNet':
        domain_list = ['clipart', 'infograph', 'painting']
        args.class_num = 40  # 345

    num_domain = len(domain_list)

    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    args.data_env = 'gpu'  # 'local'
    args.seed = 2022
    fix_random_seed(args.seed)
    tr.backends.cudnn.deterministic = True

    args.method = 'single'
    mdl_path = 'outputs/models/'
    args.output = mdl_path + dset_n + '/source/'
    print(dset_n, args.method)
    print(args)

    args.local_dir = r'/mnt/ssd2/wenz/code/NT-Benchmark/NT_SSDA/'
    args.result_dir = 'results/source/'
    my_log = LogRecord(args)
    my_log.log_init()
    my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

    acc_all = []
    for s in range(num_domain):
        args.src = s
        info_str = '\n========================== Within domain ' + domain_list[s] + ' =========================='
        print(info_str)
        my_log.record(info_str)
        args.log = my_log

        args.name_src = domain_list[s]
        args.output_dir_src = osp.join(args.output, args.name_src)
        create_folder(args.output_dir_src, args.data_env, args.local_dir)

        folder = "checkpoint/"
        args.s_dset_path = folder + args.dset + '/' + domain_list[args.src] + '_list.txt'
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

