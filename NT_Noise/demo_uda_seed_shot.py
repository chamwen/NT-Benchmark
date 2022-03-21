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
from scipy.spatial.distance import cdist

from utils import network, loss
from utils.dataloader import read_seed_single
from utils.utils import lr_scheduler, fix_random_seed, op_copy, cal_acc_noimg


def data_load(X, y, args):
    dset_loaders = {}
    train_bs = args.batch_size

    sample_idx = tr.from_numpy(np.arange(len(y))).long()
    data_tar = Data.TensorDataset(X, y, sample_idx)
    data_test = Data.TensorDataset(X, y, sample_idx)

    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True)
    dset_loaders["Target"] = Data.DataLoader(data_test, batch_size=train_bs * 3, shuffle=False)
    return dset_loaders


def train_target(args):
    X_tar, y_tar = read_seed_single(args, args.tar)
    dset_loaders = data_load(X_tar, y_tar, args)

    # base network feature extract
    netF, netC = network.backbone_net(args, args.bottleneck)

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(tr.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
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

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            mem_label = obtain_label(dset_loaders["Target"], netF, netC, args)
            mem_label = tr.from_numpy(mem_label).cuda()
            netF.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        features_test = netF(inputs_test)
        _, outputs_test = netC(features_test)

        # # loss definition
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
            acc_t_te, _ = cal_acc_noimg(dset_loaders["Target"], netF, netC)
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, iter_num, max_iter, acc_t_te)
            print(log_str)
            netF.train()

    if iter_num == max_iter:
        print('{}, TL Acc = {:.2f}%'.format(args.task_str, acc_t_te))
        return acc_t_te


def obtain_label(loader, netF, netC, args):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netF(inputs)
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
    print(log_str)

    return pred_label.astype('int')


if __name__ == '__main__':

    data_name = 'SEED'
    if data_name == 'SEED': chn, class_num, trial_num = 62, 3, 3394
    focus_domain_idx = [0, 1, 2]
    # focus_domain_idx = np.arange(15)
    domain_list = ['S' + str(i) for i in focus_domain_idx]
    num_domain = len(domain_list)

    args = argparse.Namespace(bottleneck=64, lr=0.01, lr_decay1=0.1, lr_decay2=1.0, ent=True,
                              gent=True, threshold=0, cls_par=0.3, ent_par=1.0, epsilon=1e-05,
                              layer='wn', N=num_domain, chn=chn, class_num=class_num, distance='cosine')

    args.dset = data_name
    args.method = 'SHOT'
    args.backbone = 'ShallowNet'
    args.batch_size = 32
    args.interval = 2
    args.max_epoch = 5
    args.input_dim = 310
    args.norm = 'zscore'
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'

    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    args.data_env = 'gpu'  # 'local'
    args.seed = 2022
    fix_random_seed(args.seed)
    tr.backends.cudnn.deterministic = True

    noise_list = np.linspace(0, 100, 11).tolist()
    num_test = len(noise_list)
    acc_all = np.zeros(num_test)
    s, t = 0, 1
    for ns in range(num_test):
        args.noise_rate = np.round(noise_list[ns] / 100, 2)
        dset_n = args.dset + '_' + str(args.noise_rate)
        print(dset_n, args.method)
        info_str = '\nnoise %s: %s --> %s' % (str(noise_list[ns]), domain_list[s], domain_list[t])
        print(info_str)
        args.src, args.tar = focus_domain_idx[s], focus_domain_idx[t]
        args.task_str = domain_list[s] + '_' + domain_list[t]

        mdl_path = 'outputs/models/'
        args.output_src = mdl_path + dset_n + '/source/'
        args.name_src = domain_list[s]
        args.output_dir_src = osp.join(args.output_src, args.name_src)
        print(args)

        acc_all[ns] = train_target(args)
    print('\nSub acc: ', np.round(acc_all, 3))
    print('Avg acc: ', np.round(np.mean(acc_all), 3))

    acc_sub_str = str(np.round(acc_all, 3).tolist())
    acc_mean_str = str(np.round(np.mean(acc_all), 3).tolist())

