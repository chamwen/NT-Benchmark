# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import numpy as np
import argparse
import torch as tr
import torch.optim as optim
import torch.utils.data as Data
import os.path as osp
import os
from utils import network, loss, utils
from utils.loss import CELabelSmooth
from utils.LogRecord import LogRecord
from utils.dataloader import read_seed_single, obtain_train_val_source
from utils.utils import create_folder, lr_scheduler, fix_random_seed, op_copy, add_label_noise_noimg


def data_load(X, y, args):
    dset_loaders = {}
    train_bs = args.batch_size
    tr.manual_seed(args.seed)
    trial_ins_num = args.trial

    if args.noise_rate > 0:
        y = add_label_noise_noimg(y, args.seed, args.class_num, args.noise_rate)

    id_train, id_val = obtain_train_val_source(y, trial_ins_num, args.validation)
    source_tr = Data.TensorDataset(X[id_train, :], y[id_train])
    dset_loaders['source_tr'] = Data.DataLoader(source_tr, batch_size=train_bs, shuffle=True, drop_last=True)

    source_te = Data.TensorDataset(X[id_val, :], y[id_val])
    dset_loaders['source_te'] = Data.DataLoader(source_te, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    return dset_loaders


def train_source(args):  # within validation
    X_src, y_src = read_seed_single(args, args.src)
    dset_loaders = data_load(X_src, y_src, args)

    netF, netC = network.backbone_net(args, args.bottleneck)
    netF.load_state_dict(tr.load(args.mdl_init_dir + 'netF.pt'))
    netC.load_state_dict(tr.load(args.mdl_init_dir + 'netC.pt'))

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 0.1}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])  # source_tr：80个
    interval_iter = max_iter // 10
    args.max_iter = max_iter
    iter_num = 0

    netF.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders['source_tr'])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

        _, outputs_source = netC(netF(inputs_source))
        classifier_loss = CELabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source,
                                                                                         labels_source)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netC.eval()

            acc_s_te, _ = utils.cal_acc_noimg(dset_loaders['source_te'], netF, netC)
            log_str = 'Task: {}, Iter:{}/{}; Val_acc = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.log.record(log_str)
            print(log_str)

            if acc_s_te >= acc_init:  # 返回验证集上最好的acc，保存对应模型
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netC.train()

    tr.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    tr.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return acc_s_te


if __name__ == '__main__':

    data_name = 'SEED'
    if data_name == 'SEED': chn, class_num, trial_num = 62, 3, 3394
    focus_domain_idx = [0, 1, 2]
    # focus_domain_idx = np.arange(15)
    domain_list = ['S' + str(i) for i in focus_domain_idx]
    num_domain = len(domain_list)

    args = argparse.Namespace(bottleneck=64, lr=0.01, epsilon=1e-05, layer='wn',
                              smooth=0, chn=chn, trial=trial_num,
                              N=num_domain, class_num=class_num)
    args.dset = data_name
    args.method = 'single'
    args.backbone = 'ShallowNet'
    args.batch_size = 32  # 32
    args.max_epoch = 50
    args.input_dim = 310
    args.norm = 'zscore'
    args.validation = 'random'
    args.mdl_init_dir = 'outputs/mdl_init/' + args.dset + '/'
    args.noise_rate = 0
    dset_n = args.dset + '_' + str(args.noise_rate)

    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    args.data_env = 'gpu'  # 'local'
    args.seed = 2022
    fix_random_seed(args.seed)
    tr.backends.cudnn.deterministic = True

    mdl_path = 'outputs/models/'
    args.output = mdl_path + dset_n + '/source/'
    print(dset_n, args.method)
    print(args)

    args.local_dir = r'/mnt/ssd2/wenz/code/NT-Benchmark/NT_UDA/'
    args.result_dir = 'results/source/'
    my_log = LogRecord(args)
    my_log.log_init()
    my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

    acc_all = []
    for s in range(num_domain):
        args.src = focus_domain_idx[s]
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
