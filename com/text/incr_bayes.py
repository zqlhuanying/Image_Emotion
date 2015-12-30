# encoding: utf-8
from __future__ import division
import numpy as np
from sklearn.naive_bayes import MultinomialNB

__author__ = 'root'
__date__ = '15-12-27'


class IncrBayes(MultinomialNB):
    """
    增量式贝叶斯
    """
    def __init__(self, alpha=0.01):
        super(IncrBayes, self).__init__(alpha)

    def update(self, c_pred):
        """
        更新类别概率和条件概率
        :param c_pred:
        :return:
        """
        self.update_class_pro(c_pred)

    def update_class_pro(self, c_pred):
        # todo
        # 暂时未考虑 log 的影响
        """
        更新类别概率
        :param c_pred:
        :return:
        """
        n_classes = len(self.classes_)
        n_samples = np.sum(self.class_count_)
        # 系数
        a = (n_classes + n_samples) / (1 + n_classes + n_samples)
        # 相加项
        b = 1 / (1 + n_classes + n_samples)
        # 相加矩阵
        b_matrix = [b if c == c_pred else 0 for c in self.classes_]

        return self.class_log_prior_ * a + np.array(b_matrix).reshape(-1, 1)
