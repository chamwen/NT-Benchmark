# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import sklearn
import scipy
from cvxopt import matrix, solvers
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
import numpy as np
import scipy.io
import scipy.linalg
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import warnings

warnings.filterwarnings('ignore')
solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')
solvers.options['show_progress'] = False


def baseline_SVM(X_source, Y_source, X_target, Y_target):
    # hyper-parameter:
    C_list = [0.001, 0.01, 0.1, 1, 10]

    # search the best parameter:
    model = SVC(kernel='linear')
    param_grid = {"C": C_list}
    gridsearch = GridSearchCV(model, param_grid=param_grid, cv=3)
    gridsearch.fit(X_source, Y_source)
    best_model = gridsearch.best_estimator_

    # training the model:
    best_model.fit(X_source, Y_source)
    result = best_model.predict(X_target)
    return result


def baseline_KNN(X_source, Y_source, X_target, Y_target):
    # hyper-parameter:
    K_list = [1, 3, 5, 7, 9]

    # search the best parameter:
    model = KNeighborsClassifier(weights='uniform')
    param_grid = {"n_neighbors": K_list}
    gridsearch = GridSearchCV(model, param_grid=param_grid, cv=3)
    gridsearch.fit(X_source, Y_source)
    best_model = gridsearch.best_estimator_

    # training the model:
    best_model.fit(X_source, Y_source)
    result = best_model.predict(X_target)

    return result


def kernel_kmm(ker, X1, X2, gamma):
    K = None
    if ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), None, gamma)
    return K


class KMM:
    def __init__(self, kernel_type='linear', gamma=1.0, B=1.0, eps=None):
        '''
        Initialization function
        :param kernel_type: 'linear' | 'rbf'
        :param gamma: kernel bandwidth for rbf kernel
        :param B: bound for beta
        :param eps: bound for sigma_beta
        '''
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        '''
        Fit source and target using KMM (compute the coefficients)
        :param Xs: ns * dim
        :param Xt: nt * dim
        :return: Coefficients (Pt / Ps) value vector (Beta in the paper)
        '''
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps == None:
            self.eps = self.B / np.sqrt(ns)
        K = kernel_kmm(self.kernel_type, Xs, None, self.gamma)
        kappa = np.sum(kernel_kmm(self.kernel_type, Xs, Xt, self.gamma) * float(ns) / float(nt), axis=1)

        K = matrix(K.astype(np.double))
        kappa = matrix(kappa.astype(np.double))
        G = matrix(np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)])
        h = matrix(np.r_[ns * (1 + self.eps), ns * (self.eps - 1), self.B * np.ones((ns,)), np.zeros((ns,))])

        sol = solvers.qp(K, -kappa, G, h)
        beta = np.array(sol['x'])
        return beta


class CORAL:
    def __init__(self):
        super(CORAL, self).__init__()

    def fit(self, Xs, Xt):
        '''
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        '''
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                         scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        Xs_new = np.real(np.dot(Xs, A_coral))
        return Xs_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Perform CORAL, then predict using 1NN classifier
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted labels of target domain
        '''
        Xs_new = self.fit(Xs, Xt)

        y_pred = baseline_SVM(Xs_new, Ys, Xt, Yt)
        acc = accuracy_score(Yt, y_pred) * 100

        return acc, y_pred


def kernel_jda(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class JDA:
    def __init__(self, kernel_type='primal', mmd_type='djp-mmd', dim=30, lamb=1, gamma=1, T=5):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        '''
        self.kernel_type = kernel_type
        self.mmd_type = mmd_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        X = np.hstack((Xs.T, Xt.T))
        X = np.dot(X, np.diag(1. / np.linalg.norm(X, axis=0)))
        m, n = X.shape  # 800, 2081
        ns, nt = len(Xs), len(Xt)

        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))

        Y_tar_pseudo = None
        list_acc = []
        for itr in range(self.T):

            N = 0
            n = ns + nt
            e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
            M0 = e * e.T * C
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(1, C + 1):
                    e = np.zeros((n, 1))
                    tt = Ys == c
                    if len(Ys[np.where(Ys == c)]) != 0:
                        e[np.where(tt == True)] = 1 / len(Ys[np.where(Ys == c)])
                    else:
                        e[np.where(tt == True)] = 0
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    if len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)]) != 0:
                        e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                    else:
                        e[tuple(inds)] = 0
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)
            M = M0 + N
            M = M / np.linalg.norm(M, 'fro')

            K = kernel_jda(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            Y_tar_pseudo = baseline_SVM(Xs_new, Ys, Xt_new, Yt)

            acc = accuracy_score(Yt, Y_tar_pseudo) * 100
            list_acc.append(acc)
        return list_acc[-1]
