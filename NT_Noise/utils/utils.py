# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import os.path as osp
import os
import numpy as np
import random
import torch as tr
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
from utils.loss import Entropy
import utils.network as network
from utils.dataloader import read_seed_src_tar
from utils.data_list import ImageList, ImageList_idx


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def fix_random_seed(SEED):
    tr.manual_seed(SEED)
    tr.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    tr.cuda.manual_seed_all(SEED)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def lr_scheduler_full(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def create_folder(dir_name, data_env, win_root):
    if not osp.exists(dir_name):
        os.system('mkdir -p ' + dir_name)
    if not osp.exists(dir_name):
        if data_env == 'gpu':
            os.mkdir(dir_name)
        elif data_env == 'local':
            os.makedirs(win_root + dir_name)


def image_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def create_folder(dir_name, data_env, win_root):
    if not osp.exists(dir_name):
        os.system('mkdir -p ' + dir_name)
    if not osp.exists(dir_name):
        if data_env == 'gpu':
            os.mkdir(dir_name)
        elif data_env == 'local':
            os.makedirs(win_root + dir_name)


def cal_acc_base(loader, model, flag=True, fc=None):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    feas, outputs = model(inputs)
                    outputs = fc(feas)
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    accuracy = tr.sum(tr.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy * 100


def cal_acc_img(loader, netF, netB, netC, flag=False):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    _, predict = tr.max(all_output, 1)
    accuracy = tr.sum(tr.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = tr.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    # if flag:
    #     matrix = confusion_matrix(all_label, tr.squeeze(predict).float())
    #     acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    #     aacc = acc.mean()
    #     aa = [str(np.round(i, 2)) for i in acc]
    #     acc = ' '.join(aa)
    #     return aacc, acc
    # else:
    #     return accuracy * 100, mean_ent

    return accuracy * 100, mean_ent


def cal_acc_noimg(loader, netF, netC):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0].cuda()
            labels = data[1].float()
            _, outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    accuracy = tr.sum(tr.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = tr.mean(Entropy(all_output)).cpu().data.item()

    return accuracy * 100, mean_ent


def save_fea_base(args):
    if args.dset in ['SEED', 'blob', 'moon']:
        Xs, Ys, Xt, Yt = read_seed_src_tar(args)
        dset_loaders = data_load_noimg(Xs, Ys, Xt, Yt, args)
    else:
        dset_loaders = data_load_img(args)
    source_data = dset_loaders['Source']
    target_data = dset_loaders['Target']

    # set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = osp.join(args.output_dir, "sourceF" + "_" + str(args.repeat) + ".pt")
    netF.load_state_dict({k.replace('module.', ''): v for k, v in tr.load(args.modelpath).items()})
    args.modelpath = osp.join(args.output_dir, "sourceB" + "_" + str(args.repeat) + ".pt")
    netB.load_state_dict(tr.load(args.modelpath))
    args.modelpath = osp.join(args.output_dir, "sourceC" + "_" + str(args.repeat) + ".pt")
    netC.load_state_dict(tr.load(args.modelpath))

    netF.eval()
    netB.eval()
    netC.eval()

    start_test = True
    with tr.no_grad():
        iter_train = iter(source_data)
        for i in range(len(source_data)):
            data = iter_train.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            fea = netB(netF(inputs))
            _, outputs = netC(fea)
            if start_test:
                source_output = outputs.float().cpu()
                source_label = labels.float().cpu()
                source_fea = fea.float().cpu()
                start_test = False
            else:
                source_output = tr.cat((source_output, outputs.float().cpu()), 0)
                source_label = tr.cat((source_label, labels.float().cpu()), 0)
                source_fea = tr.cat((source_fea, fea.float().cpu()), 0)

    X_source = source_fea.detach().numpy()
    y_source = source_label.detach().numpy()

    start_test = True
    with tr.no_grad():
        iter_test = iter(target_data)
        for i in range(len(target_data)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            fea = netB(netF(inputs))
            _, outputs = netC(fea)
            if start_test:
                target_output = outputs.float().cpu()
                target_label = labels.float().cpu()
                target_fea = fea.float().cpu()
                start_test = False
            else:
                target_output = tr.cat((target_output, outputs.float().cpu()), 0)
                target_label = tr.cat((target_label, labels.float().cpu()), 0)
                target_fea = tr.cat((target_fea, fea.float().cpu()), 0)

    X_target = target_fea.detach().numpy()
    y_target = target_label.detach().numpy()

    # output_source = source_output.detach().numpy()
    # output_target = target_output.detach().numpy()
    # weight = netC.fc.weight.cpu().detach().permute(1, 0).numpy()
    # bias = netC.fc.bias.cpu().detach().view(-1).numpy()
    # ACC = test_target_img(args)

    save_path = osp.join(args.fea_dir, args.task_str + "_" + str(args.repeat) + ".npz")
    np.savez(save_path, X_source=X_source, y_source=y_source, X_target=X_target, y_target=y_target)


def test_target_img(args):
    dset_loaders = data_load_img(args)
    # set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = osp.join(args.output_dir, "sourceF" + "_" + str(args.repeat) + ".pt")
    netF.load_state_dict({k.replace('module.', ''): v for k, v in tr.load(args.modelpath).items()})
    args.modelpath = osp.join(args.output_dir, "sourceB" + "_" + str(args.repeat) + ".pt")
    netB.load_state_dict(tr.load(args.modelpath))
    args.modelpath = osp.join(args.output_dir, "sourceC" + "_" + str(args.repeat) + ".pt")
    netC.load_state_dict(tr.load(args.modelpath))

    netF.eval()
    netB.eval()
    netC.eval()

    acc, _ = cal_acc_img(dset_loaders['Target'], netF, netB, netC, False)
    log_str = 'Test: {}, Task: {}, Acc = {:.2f}%'.format(args.trte, args.task_str, acc)
    print(log_str)

    return acc


def data_load_img(args):
    # prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.test_dset_path).readlines()

    # only add for source domain
    if args.noise_rate > 0:
        txt_src = add_label_noise_img(args, txt_src)

    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        tr.manual_seed(args.seed)
        tr_txt, te_txt = tr.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        tr.manual_seed(args.seed)
        _, te_txt = tr.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    # for DAN/DANN/CDAN/MCC
    dsets["source"] = ImageList(txt_src, transform=image_train())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=True)
    dsets["target"] = ImageList(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True)

    # for DNN
    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=True)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=False,
                                           num_workers=args.worker, drop_last=False)

    # for generating feature
    dsets["Source"] = ImageList(txt_src, transform=image_test())
    dset_loaders["Source"] = DataLoader(dsets["Source"], batch_size=train_bs * 3, shuffle=False,
                                        num_workers=args.worker, drop_last=False)
    dsets["Target"] = ImageList(txt_tar, transform=image_test())
    dset_loaders["Target"] = DataLoader(dsets["Target"], batch_size=train_bs * 3, shuffle=False,
                                        num_workers=args.worker, drop_last=False)

    return dset_loaders


def data_load_noimg(Xs, Ys, Xt, Yt, args):
    dset_loaders = {}
    train_bs = args.batch_size

    if args.noise_rate > 0:
        Ys = add_label_noise_noimg(Ys, args.seed, args.class_num, args.noise_rate)

    args.validation = 'random'
    src_idx = np.arange(len(Ys.numpy()))
    if args.validation == 'random':
        num_train = int(0.9 * len(src_idx))
        tr.manual_seed(args.seed)
        id_train, id_val = tr.utils.data.random_split(src_idx, [num_train, len(src_idx) - num_train])

    source_tr = Data.TensorDataset(Xs[id_train, :], Ys[id_train])
    source_te = Data.TensorDataset(Xs[id_val, :], Ys[id_val])
    data_tar = Data.TensorDataset(Xt, Yt)
    data_src = Data.TensorDataset(Xs, Ys)

    # for DAN/DANN/CDAN/MCC
    dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True, drop_last=True)

    # for DNN
    dset_loaders["source_tr"] = Data.DataLoader(source_tr, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["source_te"] = Data.DataLoader(source_te, batch_size=train_bs, shuffle=False, drop_last=False)

    # for generating feature
    dset_loaders["Source"] = Data.DataLoader(data_src, batch_size=train_bs * 3, shuffle=False, drop_last=False)
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    return dset_loaders


def add_label_noise_img(args, txt_list):
    txt_path_list = [i.split('==')[0] for i in txt_list]
    lbl_list = [int(i.split('==')[1]) for i in txt_list]

    random.seed(args.seed)
    idx_shuffle = random.sample(np.arange(len(lbl_list)).tolist(), int(args.noise_rate * len(lbl_list)))
    idx_shuffle.sort()
    class_list = np.arange(args.class_num)
    lbl_list_new = lbl_list.copy()
    for i in range(len(idx_shuffle)):
        class_list_tmp = class_list.copy().tolist()
        class_list_tmp.remove(lbl_list[idx_shuffle[i]])
        random.seed(args.seed + i)
        lbl_list_new[idx_shuffle[i]] = random.sample(class_list_tmp, 1)[0]

    txt_list_new = []
    for i in range(len(lbl_list)):
        txt_list_new.append(txt_path_list[i] + '==' + str(lbl_list_new[i]) + '\n')

    return txt_list_new


def add_label_noise_noimg(Y_raw, seed, class_num, noise_rate):
    if tr.is_tensor(Y_raw):
        Y_raw_np = Y_raw.clone().numpy()
    else:
        Y_raw_np = Y_raw.copy()

    random.seed(seed)
    idx_shuffle = random.sample(np.arange(len(Y_raw_np)).tolist(), int(noise_rate * len(Y_raw_np)))
    idx_shuffle.sort()

    class_list = np.arange(class_num)
    Y_new = Y_raw_np.copy()
    for i in range(len(idx_shuffle)):
        class_list_tmp = class_list.copy().tolist()
        class_list_tmp.remove(Y_raw_np[idx_shuffle[i]])
        random.seed(seed + i)
        Y_new[idx_shuffle[i]] = random.sample(class_list_tmp, 1)[0]
    if tr.is_tensor(Y_raw):
        Y_new = tr.from_numpy(Y_new).long()

    return Y_new


def data_load_img_ssda(args):
    # prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.test_dset_path).readlines()

    # only add for source domain in SSDA
    if args.noise_rate > 0:
        txt_src = add_label_noise_img(args, txt_src)

    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        tr.manual_seed(args.seed)
        txt_src_tr, txt_src_te = tr.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        tr.manual_seed(args.seed)
        _, txt_src_te = tr.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        txt_src_tr = txt_src

    lbl_rate = args.tar_lbl_rate / 100
    random.seed(args.seed)
    idx_train = random.sample(np.arange(len(txt_tar)).tolist(), int(lbl_rate * len(txt_tar)))
    idx_train.sort()
    idx_test = [i for i in range(len(txt_tar)) if i not in idx_train]
    txt_tar_tr = np.array(txt_tar)[idx_train]
    txt_tar_te = np.array(txt_tar)[idx_test]

    # for DAN/DANN/CDAN/MCC
    dsets["source"] = ImageList(txt_src, transform=image_train())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=True)
    dsets["target_tr"] = ImageList(txt_tar_tr, transform=image_train())
    dset_loaders["target_tr"] = DataLoader(dsets["target_tr"], batch_size=args.bz_tar_tr, shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=True)
    dsets["target_te"] = ImageList(txt_tar_te, transform=image_test())
    dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=args.bz_tar_te, shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=True)

    # for DNN, S+T, finetune
    dsets["source_tr"] = ImageList(txt_src_tr, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=True)
    dsets["source_te"] = ImageList(txt_src_te, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=False,
                                           num_workers=args.worker, drop_last=False)

    # for generating feature
    dsets["Source"] = ImageList(txt_src, transform=image_train())
    dset_loaders["Source"] = DataLoader(dsets["Source"], batch_size=train_bs * 3, shuffle=False,
                                        num_workers=args.worker, drop_last=False)
    dsets["Target"] = ImageList(txt_tar_te, transform=image_test())
    dset_loaders["Target"] = DataLoader(dsets["Target"], batch_size=train_bs * 3, shuffle=False,
                                        num_workers=args.worker,
                                        drop_last=False)

    return dset_loaders


def data_load_noimg_ssda(Xs, Ys, Xt, Yt, args):
    dset_loaders = {}
    train_bs = args.batch_size

    if args.noise_rate > 0:
        Ys = add_label_noise_noimg(Ys, args.seed, args.class_num, args.noise_rate)

    args.validation = 'random'
    src_idx = np.arange(len(Ys.numpy()))
    if args.validation == 'random':
        num_train = int(0.9 * len(src_idx))
        tr.manual_seed(args.seed)
        id_train, id_val = tr.utils.data.random_split(src_idx, [num_train, len(src_idx) - num_train])

    source_tr = Data.TensorDataset(Xs[id_train, :], Ys[id_train])
    source_te = Data.TensorDataset(Xs[id_val, :], Ys[id_val])

    idx_train, idx_test = get_idx_ssda_seed(Yt, args.tar_lbl_rate)
    target_tr = Data.TensorDataset(Xt[idx_train, :], Yt[idx_train])
    target_te = Data.TensorDataset(Xt[idx_test, :], Yt[idx_test])

    data_tar = Data.TensorDataset(Xt, Yt)
    data_src = Data.TensorDataset(Xs, Ys)

    # for DAN/DANN/CDAN/MCC
    dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["target_tr"] = Data.DataLoader(target_tr, batch_size=args.bz_tar_tr, shuffle=True, drop_last=True)
    dset_loaders["target_te"] = Data.DataLoader(target_te, batch_size=args.bz_tar_te, shuffle=True, drop_last=True)

    # for DNN
    dset_loaders["source_tr"] = Data.DataLoader(source_tr, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["source_te"] = Data.DataLoader(source_te, batch_size=train_bs, shuffle=False, drop_last=False)

    # for generating feature
    dset_loaders["Source"] = Data.DataLoader(data_src, batch_size=train_bs * 3, shuffle=False, drop_last=False)
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    return dset_loaders


def save_fea_base_ssda(args):
    if args.dset in ['SEED', 'blob', 'moon']:
        Xs, Ys, Xt, Yt = read_seed_src_tar(args)
        dset_loaders = data_load_noimg_ssda(Xs, Ys, Xt, Yt, args)
    else:
        dset_loaders = data_load_img_ssda(args)
    source_data = dset_loaders['Source']
    target_data = dset_loaders['Target']

    # set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = osp.join(args.output_dir, "sourceF" + "_" + str(args.repeat) + ".pt")
    netF.load_state_dict({k.replace('module.', ''): v for k, v in tr.load(args.modelpath).items()})
    args.modelpath = osp.join(args.output_dir, "sourceB" + "_" + str(args.repeat) + ".pt")
    netB.load_state_dict(tr.load(args.modelpath))
    args.modelpath = osp.join(args.output_dir, "sourceC" + "_" + str(args.repeat) + ".pt")
    netC.load_state_dict(tr.load(args.modelpath))

    netF.eval()
    netB.eval()
    netC.eval()

    start_test = True
    with tr.no_grad():
        iter_train = iter(source_data)
        for i in range(len(source_data)):
            data = iter_train.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            fea = netB(netF(inputs))
            _, outputs = netC(fea)
            if start_test:
                source_output = outputs.float().cpu()
                source_label = labels.float().cpu()
                source_fea = fea.float().cpu()
                start_test = False
            else:
                source_output = tr.cat((source_output, outputs.float().cpu()), 0)
                source_label = tr.cat((source_label, labels.float().cpu()), 0)
                source_fea = tr.cat((source_fea, fea.float().cpu()), 0)

    X_source = source_fea.detach().numpy()
    y_source = source_label.detach().numpy()

    start_test = True
    with tr.no_grad():
        iter_test = iter(target_data)
        for i in range(len(target_data)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            fea = netB(netF(inputs))
            _, outputs = netC(fea)
            if start_test:
                target_output = outputs.float().cpu()
                target_label = labels.float().cpu()
                target_fea = fea.float().cpu()
                start_test = False
            else:
                target_output = tr.cat((target_output, outputs.float().cpu()), 0)
                target_label = tr.cat((target_label, labels.float().cpu()), 0)
                target_fea = tr.cat((target_fea, fea.float().cpu()), 0)

    X_target = target_fea.detach().numpy()
    y_target = target_label.detach().numpy()

    if args.dset in ['SEED', 'blob', 'moon']:
        idx_train, idx_test = get_idx_ssda_seed(y_target, args.tar_lbl_rate)
    else:
        lbl_rate = args.tar_lbl_rate / 100
        random.seed(args.seed)
        idx_train = random.sample(np.arange(len(y_target)).tolist(), int(lbl_rate * len(y_target)))
        idx_train.sort()
        idx_test = [i for i in range(len(y_target)) if i not in idx_train]

    Xt_tr, Yt_tr = X_target[idx_train, :], y_target[idx_train]
    Xt_te, Yt_te = X_target[idx_test, :], y_target[idx_test]

    # output_source = source_output.detach().numpy()
    # output_target = target_output.detach().numpy()
    # weight = netC.fc.weight.cpu().detach().permute(1, 0).numpy()
    # bias = netC.fc.bias.cpu().detach().view(-1).numpy()
    # ACC = test_target_img(args)

    save_path = osp.join(args.fea_dir, args.task_str + "_" + str(args.repeat) + ".npz")
    np.savez(save_path, X_source=X_source, y_source=y_source, X_target_tr=Xt_tr, y_target_tr=Yt_tr,
             X_target_te=Xt_te, y_target_te=Yt_te)


def get_idx_ssda_seed(y, tar_lbl_rate):
    if tr.is_tensor(y):
        y_raw_np = y.clone().numpy()
    else:
        y_raw_np = y.copy()

    class_num = len(np.unique(y_raw_np))
    lbl_rate = tar_lbl_rate / 100
    num_select_class = int(lbl_rate * len(y_raw_np) / class_num)
    idx_c_list = [np.where(y_raw_np == c)[0] for c in range(class_num)]
    idx_tar_tr = []
    for c in range(class_num):
        idx_tar_tr.extend(idx_c_list[c][:num_select_class])
    idx_tar_tr.sort()
    idx_tar_te = [i for i in range(len(y_raw_np)) if i not in idx_tar_tr]

    return idx_tar_tr, idx_tar_te

