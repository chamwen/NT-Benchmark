# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import numpy as np
import torch as tr
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm


# =============================================================Common Network and Function==============================
# dynamic change the weight of the domain-discriminator
def calc_coeff(iter_num, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 / (1.0 + np.exp(-alpha * iter_num / max_iter)) - 1)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        y = self.fc(x)
        return x, y


# =============================================================Base Network and Function for SEED/Synth data============
class ShallowNet(nn.Module):
    def __init__(self, input_size, hidden_size, radius=10.0):
        super(ShallowNet, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.__in_features = hidden_size
        self.radius = radius

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        x = self.radius * x / (tr.norm(x, dim=1, keepdim=True) + 1e-10)

        return x


def backbone_net(args, n_hidden):
    if args.backbone == 'ShallowNet':
        netF = ShallowNet(args.input_dim, n_hidden, radius=5.0).cuda()

    netC = feat_classifier(class_num=args.class_num, bottleneck_dim=args.bottleneck, type=args.layer).cuda()

    return netF, netC


# =============================================================Base Network and Function for Vision=====================
vgg_dict = {"vgg11": models.vgg11, "vgg13": models.vgg13, "vgg16": models.vgg16, "vgg19": models.vgg19,
            "vgg11bn": models.vgg11_bn, "vgg13bn": models.vgg13_bn, "vgg16bn": models.vgg16_bn,
            "vgg19bn": models.vgg19_bn}

res_dict = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50,
            "resnet101": models.resnet101, "resnet152": models.resnet152, "resnext50": models.resnext50_32x4d,
            "resnext101": models.resnext101_32x8d}


class VGGBase(nn.Module):
    def __init__(self, vgg_name):
        super(VGGBase, self).__init__()
        model_vgg = vgg_dict[vgg_name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_vgg.classifier[i])
        self.in_features = model_vgg.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y


class Discriminator(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ln1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.ln2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = self.ln2(self.bn(x))
        y = tr.sigmoid(x)
        return y


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]
