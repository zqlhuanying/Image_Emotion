# encoding: utf-8
from neupy import algorithms
from sklearn import datasets, metrics
from sklearn.cross_validation import train_test_split

from com.image.pnn import PNN

__author__ = 'zql'
__date__ = '2016/06/14'


def test0():
    """
    daemon
    :return:
    """
    dataset = datasets.load_digits()
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size=0.7)
    nw = algorithms.PNN(std=10, verbose=False)
    nw.train(x_train, y_train)
    result = nw.predict(x_test)
    print metrics.accuracy_score(y_test, result)


def test1():
    """
    数字型类别
    :return:
    """
    dataset = datasets.load_digits()
    n_samples = dataset.data.shape[0]
    ratio = 0.7
    x_train, y_train = dataset.data[0:n_samples * ratio, :], dataset.target[0:n_samples * ratio]
    x_test, y_test = dataset.data[n_samples * ratio:, :], dataset.target[n_samples * ratio:]
    nw = algorithms.PNN(std=10, verbose=False)
    nw.train(x_train, y_train)
    result = nw.predict(x_test)
    print metrics.accuracy_score(y_test, result)


def test2():
    """
    中文类别
    :return:
    """
    dataset = datasets.load_digits()
    n_samples = dataset.data.shape[0]
    ratio = 0.7
    x_train, y_train = dataset.data[0:n_samples * ratio, :], dataset.target[0:n_samples * ratio]
    x_test, y_test = dataset.data[n_samples * ratio:, :], dataset.target[n_samples * ratio:]
    class_label = ("anger", "disgust", "fear", "happiness", "like", "sadness", "surprise",
                   "anger0", "anger1", "anger2")
    y_train, y_test = y_train.astype("str"), y_test.astype("str")
    for y in (y_train, y_test):
        for i in range(y.shape[0]):
            y[i] = class_label[int(y[i])]
    nw = PNN(std=10)
    nw.get_classificator(x_train, y_train)
    result = nw.predict(x_test)
    print metrics.accuracy_score(y_test, result)

if __name__ == "__main__":
    test1()
    test2()
