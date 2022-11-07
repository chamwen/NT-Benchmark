# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import argparse
import os
import os.path as osp
import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optim
import random
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from utils import network, loss
from utils.data_list import ImageList_idx
from utils.utils import cal_acc_img, op_copy, lr_scheduler
from utils.utils import image_train, image_test
from utils.LogRecord import LogRecord


def data_load(args):
    # prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()

    lbl_rate = args.tar_lbl_rate / 100
    random.seed(args.seed)
    idx_train = random.sample(np.arange(len(txt_tar)).tolist(), int(lbl_rate * len(txt_tar)))
    idx_train.sort()
    idx_test = [i for i in range(len(txt_tar)) if i not in idx_train]
    txt_tar_te = np.array(txt_tar)[idx_test]

    dsets["target_te"] = ImageList_idx(txt_tar_te, transform=image_train())
    dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["Target"] = ImageList_idx(txt_tar, transform=image_test())
    dset_loaders["Target"] = DataLoader(dsets["Target"], batch_size=train_bs * 3, shuffle=False,
                                        num_workers=args.worker, drop_last=False)

    return dset_loaders


def train_target(args):
    dset_loaders = data_load(args)

    # set base network
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + "/sourceF" + "_" + str(args.repeat) + ".pt"
    netF.load_state_dict(tr.load(modelpath))
    modelpath = args.output_dir_src + "/sourceB" + "_" + str(args.repeat) + ".pt"
    netB.load_state_dict(tr.load(modelpath))
    modelpath = args.output_dir_src + "/sourceC" + "_" + str(args.repeat) + ".pt"
    netC.load_state_dict(tr.load(modelpath))
    netC.eval()

    for k, v in netC.named_parameters():
        v.requires_grad = False

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

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target_te"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target_te"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            mem_label = obtain_label(dset_loaders["Target"], netF, netB, netC, args)
            mem_label = tr.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        features_test = netB(netF(inputs_test))
        _, outputs_test = netC(features_test)

        # loss definition
        if args.cls_par > 0:
            pred = mem_label[tar_idx].long()
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
        else:
            classifier_loss = tr.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = tr.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = tr.sum(msoftmax * tr.log(msoftmax + args.epsilon))
                entropy_loss += gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            acc_t_te, _ = cal_acc_img(dset_loaders["Target"], netF, netB, netC)
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, iter_num, max_iter, acc_t_te)

            args.out_file.write(log_str)
            args.out_file.flush()
            print(log_str)
            netF.train()
            netB.train()

    if iter_num == max_iter:
        print('{}, TL Acc = {:.2f}%'.format(args.task_str, acc_t_te))
        return acc_t_te


def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            _, outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = tr.cat((all_fea, feas.float().cpu()), 0)
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = tr.sum(-all_output * tr.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = tr.max(all_output, 1)

    accuracy = tr.sum(tr.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = tr.cat((all_fea, tr.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / tr.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):  # SSL
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'SSL_Acc = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write('\t\t' + log_str + '\n')
    args.out_file.flush()
    print(log_str)

    return pred_label.astype('int')


if __name__ == "__main__":

    # python image_target.py --cls_par 0.3 --s 0
    args = argparse.Namespace(batch_size=32, bottleneck=1024, classifier='bn', cls_par=0.3,
                              distance='cosine', ent=True, ent_par=1.0, epsilon=1e-05,
                              gent=True, interval=2, layer='wn', lr=0.01,
                              lr_decay1=0.1, lr_decay2=1.0, max_epoch=5, s=0, t=1, threshold=0, worker=4)

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    args.data_env = 'gpu'  # 'local'

    args.seed = 2022
    tr.manual_seed(args.seed)
    tr.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # tr.backends.cudnn.deterministic = True

    args.dset = 'DomainNet'
    args.method = 'SHOT'
    args.net = 'resnet50'
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'
    args.noise_rate = 0
    dset_n = args.dset + '_' + str(args.noise_rate)
    args.tar_lbl_rate = 5  # [5, 10, ..., 50]/100

    if args.dset == 'DomainNet':
        domain_list = ['clipart', 'infograph', 'painting']
        args.class_num = 40  # 345

    num_domain = len(domain_list)
    args.method = 'SHOT'
    mdl_path = 'outputs/mdl_fea/'
    args.repeat = 0
    args.output_src = mdl_path + dset_n
    print(dset_n, args.method)
    print(args)

    args.local_dir = r'/mnt/ssd2/wenz/code/TL/TL_Understand/NT_SSDA/'
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
                args.src, args.tar = s, t
                my_log.record(info_str)
                args.log = my_log

                args.task_str = domain_list[args.src][0].upper() + domain_list[args.tar][0].upper()
                args.output_dir_src = osp.join(args.output_src, args.task_str + '_tr' + str(args.tar_lbl_rate))

                folder = "checkpoint/"
                args.t_dset_path = folder + args.dset + '/' + domain_list[args.tar] + '_list.txt'
                print(args)

                acc_all[itr_idx] = train_target(args)
    print('\nSub acc: ', np.round(acc_all, 3))
    print('Avg acc: ', np.round(np.mean(acc_all), 3))

    acc_sub_str = str(np.round(acc_all, 3).tolist())
    acc_mean_str = str(np.round(np.mean(acc_all), 3).tolist())
    args.log.record("\n==========================================")
    args.log.record(acc_sub_str)
    args.log.record(acc_mean_str)

