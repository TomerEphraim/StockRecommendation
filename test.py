import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    x = []
    x.append([0.7])
    x.append([0.2, 0.3])
    x.append([0.4, 0.5, 1.0])
    X = np.empty((len(x),), dtype=list)
    for i, v in enumerate(x):
        X[i] = x[i]
    print(X)
    var = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=1, dtype=float)
    print(var)