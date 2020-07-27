from pandas import read_csv
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
import numpy as np
from collections import Counter
from matplotlib import pyplot
from numpy import where
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from numpy import vstack
from sklearn.neighbors import LocalOutlierFactor
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn import metrics


labels = {"Buy": 0, "NotBuy": 1}
index_column_name = "Date"
value_column_name = "Close"


def lof_predict(model, trainX, testX):
    composite = vstack((trainX, testX))
    yhat = model.fit_predict(composite)
    return yhat[len(trainX):]


def load_data():
    x = []
    y = []
    for folder_name in labels.keys():
        for filename in os.listdir(folder_name):
            series = read_csv(os.path.join(folder_name, filename), header=0, parse_dates=[index_column_name],
                              index_col=0, usecols=[index_column_name, value_column_name]).squeeze()
            actual_max = series.max()
            actual_min = series.min()
            actual_diff_max_min = actual_max - actual_min
            normalized_max = 1
            normalized_min = 0
            normalized_diff_max_min = normalized_max - normalized_min
            series = normalized_diff_max_min / actual_diff_max_min * (series - actual_max) + normalized_max
            x.append(series.values.tolist())
            y.append(labels[folder_name])
    X = np.empty((len(x), ), dtype=list)
    for i, v in enumerate(x):
        X[i] = x[i]
    y = np.array(y)
    return X, y


def method_one_class_svm():
    X, Y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    max_review_length = min(map(len, X))
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length, dtype=float)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length, dtype=float)
    X_train = X_train[y_train == labels["Buy"]]
    model = OneClassSVM(gamma='scale', nu=0.5)
    model.fit(X_train)
    yhat = model.predict(X_test)
    print("accuracy: ", metrics.accuracy_score(y_test, yhat))
    y_test[y_test == labels["NotBuy"]] = -1
    y_test[y_test == labels["Buy"]] = 1
    score = f1_score(y_test, yhat, pos_label=-1)
    print('F1 Score: %.3f' % score)


def method_isolation_forest():
    X, Y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    max_review_length = min(map(len, X))
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length, dtype=float)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length, dtype=float)
    X_train = X_train[y_train == labels["Buy"]]
    model = IsolationForest(contamination=0.5)
    model.fit(X_train)
    yhat = model.predict(X_test)
    print("accuracy: ", metrics.accuracy_score(y_test, yhat))
    y_test[y_test == labels["NotBuy"]] = -1
    y_test[y_test == labels["Buy"]] = 1
    score = f1_score(y_test, yhat, pos_label=-1)
    print('F1 Score: %.3f' % score)
    # for i, v in enumerate(X_test):
    #     # res = model.predict(v)
    #     print(v)
    #     # pyplot.plot(v)
    #     # pyplot.title("Buy" if res == labels["Buy"] else "NotBuy")
    #     # pyplot.show()


# def method_mcd():
#     X, Y = load_data()
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
#     max_review_length = max(map(len, X))
#     X_train = sequence.pad_sequences(X_train, maxlen=max_review_length, dtype=float)
#     X_test = sequence.pad_sequences(X_test, maxlen=max_review_length, dtype=float)
#     X_train = X_train[y_train == labels["Buy"]]
#     model = EllipticEnvelope(contamination=0.5)
#     model.fit(X_train)
#     yhat = model.predict(X_test)
#     y_test[y_test == labels["NotBuy"]] = -1
#     y_test[y_test == labels["Buy"]] = 1
#     score = f1_score(y_test, yhat, pos_label=-1)
#     print('F1 Score: %.3f' % score)


def method_lof():
    X, Y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    max_review_length = min(map(len, X))
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length, dtype=float)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length, dtype=float)
    X_train = X_train[y_train == labels["Buy"]]
    model = LocalOutlierFactor(contamination=0.5)
    yhat = lof_predict(model, X_train, X_test)
    print("accuracy: ", metrics.accuracy_score(y_test, yhat))
    y_test[y_test == labels["NotBuy"]] = -1
    y_test[y_test == labels["Buy"]] = 1
    score = f1_score(y_test, yhat, pos_label=-1)
    print('F1 Score: %.3f' % score)


def method_LSTM():
    # np.random.seed(7)
    X, Y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    max_review_length = min(map(len, X))
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length, dtype=float)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length, dtype=float)
    # X_train = X_train[y_train == labels["Buy"]]
    # y_train = y_train[y_train == labels["Buy"]]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=256, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(filters=32, kernel_size=27, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(filters=32, kernel_size=9, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(LSTM(100, input_shape=(max_review_length, 1)))
    model.add(LSTM(100, input_shape=(max_review_length, 1), dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20)
    print(model.summary())
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == '__main__':
    method_one_class_svm()
