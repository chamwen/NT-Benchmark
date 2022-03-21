# -*- coding: utf-8 -*-
# A Survey on Negative Transfer
# https://github.com/chamwen/NT-Benchmark
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from func_transform import *

root = '/Users/NT-Benchmark/NT_UDA/data_synth/'
random_state = 2020

# Generate MOONS SYNTHETIC DATA
moons = make_moons(n_samples=600, noise=.1)
Xt, Yt = moons[0], moons[1]
data = np.concatenate([Xt, Yt.reshape(-1, 1)], axis=1)
np.savetxt(root + "moon0.csv", data, delimiter=',')
# show_data(Xt, Yt)

# Generate DSM01 - TRANSLATION
Xs, Ys = affine_transformation(Xt, Yt, "moon1", translation, 2)

# Generate DSM02 - SCALE
Xs, Ys = affine_transformation(Xt, Yt, "moon2", scale, 1.5)

# Generate DSM03 - ROTATION
Xs, Ys = affine_transformation(Xt, Yt, "moon3_15", rotation, np.pi / 12)
Xs, Ys = affine_transformation(Xt, Yt, "moon3_30", rotation, np.pi / 6)
Xs, Ys = affine_transformation(Xt, Yt, "moon3_45", rotation, np.pi / 4)

# Generate DSM04 - SHEAR
Xs, Ys = affine_transformation(Xt, Yt, "moon4_5", shear, 0.5)
Xs, Ys = affine_transformation(Xt, Yt, "moon4_10", shear, 1)
Xs, Ys = affine_transformation(Xt, Yt, "moon4_15", shear, 1.5)

# Generate DSM05 - COMBINATION
Xs, Ys = affine_transformation(Xt, Yt, "moon5", translation, 2, save=False)
Xs, Ys = affine_transformation(Xs, Yt, "moon5", rotation, np.pi / 4, save=False)
Xs, Ys = affine_transformation(Xs, Yt, "moon5", scale, 2, save=False)
Xs, Ys = affine_transformation(Xs, Yt, "moon5", shear, 1.0, save=True)

# Generate DSM06 - SKEWED DISTRIBUTION
cls = Yt == 0  # init
ind1 = np.squeeze(np.nonzero(cls))
ind2 = np.squeeze(np.nonzero(np.bitwise_not(cls)))
Xa, Ya = Xt[ind1], Yt[ind1]
Xb, Yb = Xt[ind2], Yt[ind2]
Xa = Xa + 1.5 * Xa.std() * np.random.random(Xa.shape)  # add noise
Xs = np.concatenate((Xa, Xb), axis=0)
Ys = np.concatenate((Ya, Yb), axis=0)

idx_rand = np.arange(len(Ys))
np.random.seed(random_state)
random.seed(random_state)
random.shuffle(idx_rand)
Xs, Ys = Xs[idx_rand.tolist(), :], Ys[idx_rand.tolist()]
Xs, Ys = affine_transformation(Xs, Ys, "moon6", translation, 0)

# Generate DSM07 - NOISE
Xs = noisy(Xt, "s&p")
Xs, Ys = affine_transformation(Xs, Yt, "moon7", translation, 0)

# Generate DSM08 - OVERLAPPING
moons = make_moons(n_samples=600, noise=.33)
Xs, Ys = moons[0], moons[1]
Xs, Ys = affine_transformation(Xs, Ys, "moon8", translation, 0)

# Generate DSM09 - SUB-CLUSTERS
centers = [(-0.5, -0.5)]
samples = make_blobs(n_samples=100, centers=centers, cluster_std=[0.15], random_state=random_state, shuffle=True)
moons = make_moons(n_samples=600, noise=.1)
Xs, Ys = moons[0], moons[1]
Xs = np.concatenate((Xs, samples[0]), axis=0)
Ys = np.concatenate((Ys, samples[1]), axis=0)
Xs, Ys = affine_transformation(Xs, Ys, "moon9", translation, 0)
