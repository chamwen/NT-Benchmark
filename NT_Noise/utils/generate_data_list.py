# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import os
import sys
import random
import numpy as np
import os.path as osp

sys.path.append("..")
fix_seed = 2022


def generate(dir, use_path, txt_path, label, sample_rate=1):
    files = os.listdir(dir)
    files.sort()

    if sample_rate < 1:
        select_num = int(len(files) * sample_rate)
        raw_idx = np.arange(len(files))
        random.seed(fix_seed)
        random.shuffle(raw_idx)
        select_idx = raw_idx[:select_num].tolist()
        files = np.array(files.copy())[select_idx].tolist()
        files.sort()

    total_num = len(files)
    # print(total_num)

    listText = open(txt_path, 'a')
    num = 0
    for file in files:
        num += 1
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = use_path + file + '==' + str(int(label)) + '\n'
        if num < total_num + 1:
            listText.write(name)
    listText.close()

    return total_num


def check_class_ins_num(domain_list, folderlist):
    min_class_num_list = []
    for name in domain_list:
        print('\nreading...', name)
        txt_path = out_path_root + dset + '/' + name + '_list.txt'

        class_list = []
        for line in open(txt_path):
            class_list.append(line.split('/' + name + '/')[1].split('/')[0])

        class_list = np.array(class_list)
        class_num_list = [np.sum(class_list == cn) for cn in folderlist]
        min_class_num_list.append(min(class_num_list))
        print('min class ins_num', min(class_num_list))
    print(min_class_num_list)


if __name__ == "__main__":
    root = "/mnt/ssd2/wenz/data/"
    out_path_root = '../checkpoint/'

    dset = 'VisDA17'
    if dset == 'office':
        domain_list = ['amazon', 'dslr', 'webcam']
    if dset == 'office-home':
        domain_list = ['Art', 'Clipart', 'Product', 'RealWorld']
    if dset == 'office-caltech':
        domain_list = ['amazon', 'caltech', 'dslr', 'webcam']
    if dset == 'VisDA17':
        domain_list = ['train', 'validation']
    if dset == 'DomainNet':
        domain_list = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

    save_path = out_path_root + dset
    if not osp.exists(save_path):
        os.system('mkdir -p ' + save_path)
    if not osp.exists(save_path):
        os.mkdir(save_path)

    # 40 classes refer:
    # SENTRY: Selective entropy optimization via committee consistency
    # for unsupervised domain adaptation." ICCV. 2021.
    if dset == 'DomainNet':
        folderlist = ['airplane', 'ambulance', 'apple', 'backpack', 'banana', 'bathtub', 'bear', 'bed', 'bee',
                      'bicycle', 'bird', 'book', 'bridge', 'bus', 'butterfly', 'cake', 'calculator', 'camera', 'car',
                      'cat', 'chair', 'clock', 'cow', 'dog', 'dolphin', 'donut', 'drums', 'duck', 'elephant', 'fence',
                      'fork', 'horse', 'house', 'rabbit', 'scissors', 'sheep', 'strawberry', 'table', 'telephone',
                      'truck']
        sample_rate = 0.2  # 0.2, 0.4  20%*all_num

    for name in domain_list:
        print('\nprocessing...', name)
        data_path = root + dset + '/' + name
        txt_path = out_path_root + dset + '/' + name + '_list.txt'

        if '.DS_Store' in folderlist:
            folderlist.remove('.DS_Store')

        i = 0
        total_num = 0
        for folder in folderlist:
            use_path_a = data_path + '/' + folder + '/'
            num = generate(os.path.join(data_path, folder), use_path_a, txt_path, i, sample_rate)
            total_num = total_num + num
            i += 1
        print(name, total_num)

    print('=' * 50)
    check_class_ins_num(domain_list, folderlist)
