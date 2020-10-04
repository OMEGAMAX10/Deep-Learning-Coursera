import tensorflow as tf
import keras.backend as k
from keras.layers import RepeatVector
import numpy as np


def data_processing(corpus, values_indices, m=60, Tx=30):
    # cut the corpus into semi-redundant sequences of Tx values
    Tx = Tx
    N_values = len(set(corpus))
    np.random.seed(0)
    X = np.zeros((m, Tx, N_values), dtype=np.bool)
    Y = np.zeros((m, Tx, N_values), dtype=np.bool)
    for i in range(m):
        random_idx = np.random.choice(len(corpus) - Tx)
        corp_data = corpus[random_idx:(random_idx + Tx)]
        for j in range(Tx):
            idx = values_indices[corp_data[j]]
            if j != 0:
                X[i, j, idx] = 1
                Y[i, j - 1, idx] = 1
    Y = np.swapaxes(Y, 0, 1)
    Y = Y.tolist()
    return np.asarray(X), np.asarray(Y), N_values


def one_hot(x):
    x = k.argmax(x)
    x = tf.one_hot(x, 78)
    x = RepeatVector(1)(x)
    return x
