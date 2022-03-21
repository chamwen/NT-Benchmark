# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import argparse
import os
import random
import os.path as osp
import torch as tr
import numpy as np
import utils.network as network


def create_folder(output_dir):

    if not osp.exists(output_dir):
        os.system('mkdir -p ' + output_dir)
    if not osp.exists(output_dir):
        os.mkdir(output_dir)


if __name__ == '__main__':
    seed = 2022
    tr.manual_seed(seed)
    tr.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tr.cuda.manual_seed_all(seed)
    tr.backends.cudnn.deterministic = True
    mdl_init_dir = 'outputs/mdl_init/'
    dset_list = ['DomainNet', 'SEED', 'moon']

    ###################################################################################
    # SEED data
    args = argparse.Namespace(bottleneck=64, backbone='ShallowNet', layer='wn')
    args.input_dim = 310
    args.class_num = 3
    output_dir = osp.join(mdl_init_dir, dset_list[1])
    create_folder(output_dir)

    netF, netC = network.backbone_net(args, args.bottleneck)
    netD_full = network.AdversarialNetwork(args.bottleneck, 20).cuda()
    netD_clf = network.feat_classifier(type=args.layer, class_num=2, bottleneck_dim=args.bottleneck).cuda()

    tr.save(netF.state_dict(), osp.join(output_dir, "netF.pt"))
    tr.save(netC.state_dict(), osp.join(output_dir, "netC.pt"))
    tr.save(netD_full.state_dict(), osp.join(output_dir, "netD_full.pt"))
    tr.save(netD_clf.state_dict(), osp.join(output_dir, "netD_clf.pt"))
    print('\nfinished init of seed data...')

