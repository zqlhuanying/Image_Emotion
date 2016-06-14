# encoding: utf-8
from neupy import algorithms
from numpy import unique
import numpy as np

__author__ = 'root'
__date__ = '2016/06/14'


class PNN:
    def __init__(self, std=10):
        """
        :param std: float
                    standard deviation for PDF function, default to 0.1.
        :return:
        """
        self.pnn = algorithms.PNN(std=std, verbose=False)

    def get_classificator(self, train_datas, class_label):
        """
        获取分类器, support for non numeric label
        :param train_datas: [n_samples, n_features]
        :param class_label: [n_samples,]
        :return:
        """
        class_label = np.asarray(class_label)

        # 将中文标签转成数字类型标签
        self.classes = unique(class_label)
        for i in range(class_label.shape[0]):
            class_label[i] = np.where(self.classes == class_label[i])[0][0]

        self.pnn.train(train_datas, class_label)
        return self

    def predict(self, test_datas):
        """
        predict
        :param test_datas: [n_samples, n_features]
        :return: [n_samples,]
        """
        if self.classes is None:
            raise ValueError("Train network before predict data")

        predict_output = self.pnn.predict(test_datas)
        return self.classes[predict_output.astype("int")]
