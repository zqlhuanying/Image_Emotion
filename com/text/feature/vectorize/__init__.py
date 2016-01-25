# encoding: utf-8
from __future__ import division

from sklearn.utils.extmath import safe_sparse_dot

__author__ = 'zql'
__date__ = '16-01-24'


def tobool(X):
    """
    不管 X 中具体值是多少，将其转化成 0|1 的值
    即只关心这个特征词是否出现在样本中
    :param X: sparse matrix, [n_samples, n_features]
    :return: [n_samples, n_features]
    """
    bool_ = X.astype(bool)
    return bool_.astype(float)


def class_df(X, Y):
    """
    计算每个特征词在每个类别下的文档数
    :param X: sparse matrix, [n_samples, n_features]
    :param Y: [n_samples, n_classes]
    :return: [n_classes, n_features]
    """
    # 计算每个样本中是否包含某个特征词, [n_samples, n_features]
    is_in_samples = tobool(X)

    # 计算每个特征词属于每个类别的样本数 [n_classes, n_features]
    return safe_sparse_dot(Y.T, is_in_samples)

