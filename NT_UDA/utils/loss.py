# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import numpy as np
import torch as tr
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Sequence


def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * tr.log(input_ + epsilon)
    entropy = tr.sum(entropy, dim=1)
    return entropy


class CELabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CELabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = tr.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss


class InformationMaximizationLoss(nn.Module):
    """
    Information maximization loss.
    """

    def __init__(self):
        super(InformationMaximizationLoss, self).__init__()

    def forward(self, pred_prob, epsilon):
        softmax_out = nn.Softmax(dim=1)(pred_prob)
        ins_entropy_loss = tr.mean(Entropy(softmax_out))
        msoftmax = softmax_out.mean(dim=0)
        class_entropy_loss = tr.sum(-msoftmax * tr.log(msoftmax + epsilon))
        im_loss = ins_entropy_loss - class_entropy_loss

        return im_loss


# =============================================================DAN Function=============================================
class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""
    Args:
        kernels (tuple(tr.nn.Module)): kernel functions.
        linear (bool): whether use the linear version of DAN. Default: False

    Inputs:
        - z_s (tensor): activations from the source domain, :math:`z^s`
        - z_t (tensor): activations from the target domain, :math:`z^t`
    """

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: tr.Tensor, z_t: tr.Tensor) -> tr.Tensor:
        features = tr.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)

        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[tr.Tensor] = None,
                         linear: Optional[bool] = True) -> tr.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = tr.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix


class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix
    Args:
        sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Inputs:
        - X (tensor): input group :math:`X`

    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = tr.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: tr.Tensor) -> tr.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * tr.mean(l2_distance_square.detach())

        return tr.exp(-l2_distance_square / (2 * self.sigma_square))


# =============================================================CDANE Function===========================================
def CDANE(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        # print('None')
        op_out = tr.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = tr.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + tr.exp(-entropy)
        source_mask = tr.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = tr.ones_like(entropy)
        target_mask[0:feature.size(0) // 2] = 0
        target_weight = entropy * target_mask
        weight = source_weight / tr.sum(source_weight).detach().item() + \
                 target_weight / tr.sum(target_weight).detach().item()
        return tr.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / tr.sum(
            weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [tr.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [tr.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = tr.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


# =============================================================MCC Function=============================================
class ClassConfusionLoss(nn.Module):
    """
    The class confusion loss

    Parameters:
        - **t** Optional(float): the temperature factor used in MCC
    """

    def __init__(self, t):
        super(ClassConfusionLoss, self).__init__()
        self.t = t

    def forward(self, output: tr.Tensor) -> tr.Tensor:
        n_sample, n_class = output.shape
        softmax_out = nn.Softmax(dim=1)(output / self.t)
        entropy_weight = Entropy(softmax_out).detach()
        entropy_weight = 1 + tr.exp(-entropy_weight)
        entropy_weight = (n_sample * entropy_weight / tr.sum(entropy_weight)).unsqueeze(dim=1)
        class_confusion_matrix = tr.mm((softmax_out * entropy_weight).transpose(1, 0), softmax_out)
        class_confusion_matrix = class_confusion_matrix / tr.sum(class_confusion_matrix, dim=1)
        mcc_loss = (tr.sum(class_confusion_matrix) - tr.trace(class_confusion_matrix)) / n_class
        return mcc_loss


# =============================================================MME Function=============================================
def adentropy(out_t1, lamda):
    out_t1 = F.softmax(out_t1, dim=1)
    loss_adent = lamda * tr.mean(tr.sum(out_t1 * (tr.log(out_t1 + 1e-5)), 1))
    return loss_adent


def entropy(out_t1, lamda):
    out_t1 = F.softmax(out_t1, dim=1)
    loss_ent = -lamda * tr.mean(tr.sum(out_t1 * (tr.log(out_t1 + 1e-5)), 1))
    return loss_ent


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=0.0001,
                     power=0.75, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (- power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer


# =============================================================FixBi Function===========================================
def get_sp_loss(input, target, temp):
    criterion = nn.NLLLoss(reduction='none').cuda()
    loss = tr.mul(criterion(tr.log(1 - F.softmax(input / temp, dim=1)), target.detach()), 1).mean()
    return loss


def get_target_preds(args, pred):
    top_prob, top_label = tr.topk(F.softmax(pred, dim=1), k=1)
    top_label = top_label.squeeze().t()
    top_prob = top_prob.squeeze().t()
    top_mean, top_std = top_prob.mean(), top_prob.std()
    threshold = top_mean - args.th * top_prob
    return top_label, top_prob, threshold


def mixup_criterion_hard(pred, y_src, y_tar_pse, ratio):
    criterion = nn.CrossEntropyLoss().cuda()
    loss_fm = ratio * criterion(pred, y_src) + (1 - ratio) * criterion(pred, y_tar_pse)
    return loss_fm


def get_fixmix_loss(net, src_imgs, tar_imgs, src_labels, tar_pseudo, ratio):
    mixed_x = ratio * src_imgs + (1 - ratio) * tar_imgs
    _, mixed_pred = net(mixed_x)  # predict labels of source or target
    loss = mixup_criterion_hard(mixed_pred, src_labels.detach(), tar_pseudo.detach(), ratio)
    return loss
