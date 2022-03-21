# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from sklearn import manifold
import warnings
warnings.filterwarnings('ignore')


def noisy(samples, noise_type):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    """

    if noise_type == "gauss":
        row, col = samples.shape
        mean = 0
        var = 0.05
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = samples + gauss
        return noisy

    elif noise_type == "s&p":
        row, col = samples.shape
        amount = 0.05
        out = np.copy(samples)
        num = np.ceil(amount * samples.size)

        x1_min = np.amin(samples[:, 0])
        x1_max = np.amax(samples[:, 0])
        x2_min = np.amin(samples[:, 1])
        x2_max = np.amax(samples[:, 0])

        # Pepper mode
        coords = np.random.randint(0, samples.shape[0], int(num))

        for i in range(len(coords)):
            out[coords[i]][0] = random.uniform(x1_min, x1_max)
            out[coords[i]][1] = random.uniform(x2_min, x2_max)

        return out

    elif noise_type == "poisson":
        vals = len(np.unique(samples))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(samples * vals) / float(vals)
        return noisy

    elif noise_type == "speckle":
        row, col = samples.shape
        gauss = np.random.randn(row, col)
        gauss = gauss.reshape(row, col)
        noisy = samples + samples * gauss
        return noisy


def translation(Xs, magnitude):
    for i in range(len(Xs)):
        Xs[i][0] += magnitude
        Xs[i][1] += magnitude
    return Xs


def rotation(Xs, angle):
    # Rotate data
    # axis 1
    a = np.multiply(Xs[:, 0], np.cos(angle))
    b = np.multiply(Xs[:, 1], -1 * np.sin(angle))
    T1 = np.add(a, b)
    # axis 2
    a = np.multiply(Xs[:, 1], np.cos(angle))
    b = np.multiply(Xs[:, 0], np.sin(angle))
    T2 = np.add(a, b)
    # copy rotated data
    Xs[:, 0] = T1
    Xs[:, 1] = T2
    return Xs


def scale(Xs, scalar):
    # Apply scale
    Xs = scalar * Xs
    return Xs


def reflection(Xs, scalar):
    # applying Reflection
    Xs[:, 1] = Xs[:, 1] * (-scalar)
    return Xs


def shear(Xs, scalar):
    # Applying SHEAR
    # dim-1
    a = np.multiply(Xs[:, 0], 1)
    b = np.multiply(Xs[:, 1], 0)
    T1 = np.add(a, b)
    # dim-2
    a = np.multiply(Xs[:, 1], 1)
    b = np.multiply(Xs[:, 0], scalar)
    T2 = np.add(a, b)
    # copy data
    Xs[:, 0] = T1
    Xs[:, 1] = T2
    return Xs


def show_data_st(Xt, Yt, Xs, Ys):
    # Draw synthetic data on image
    X = np.concatenate((Xt, Xs), axis=0)
    Y = np.concatenate((Yt, Ys), axis=0)

    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=Y[:], alpha=0.4)
    plt.show()


def show_data(Xs, Ys):
    # Draw synthetic data on image
    plt.clf()
    plt.scatter(Xs[:, 0], Xs[:, 1], c=Ys[:], alpha=0.4)
    plt.show()


def affine_transformation(Xt, Yt, name, affine_method, value, save=True, plot=False):
    """
    Method to generate affine transformation
    """
    # Copy to target domain
    Xs = copy.copy(Xt)
    Ys = copy.copy(Yt)

    # generate affine transformation
    Xs = affine_method(Xs, value)

    # Save target domain
    if save:
        root = '/Users/NT-Benchmark/NT_UDA/data_synth/'
        data = np.concatenate([Xs, Ys.reshape(-1, 1)], axis=1)
        np.savetxt(root + name + ".csv", data, delimiter=',')

    if plot:
        show_data(Xs, Ys)

    return Xs, Ys
