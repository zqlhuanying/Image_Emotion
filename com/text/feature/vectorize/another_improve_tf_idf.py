# encoding: utf-8
from __future__ import division

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelBinarizer, normalize
from sklearn.utils.extmath import safe_sparse_dot

from com.text.feature import vectorize

__author__ = 'zql'
__date__ = '16-01-24'


class TfidfImproveSec(object):
    """
    第二种改进 tfidf 的方法
    """
    def __init__(self, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y):
        """
        Learn the idf vector (global term weights)
        :param X: sparse matrix, [n_samples, n_features]
                  X must be a matrix of term counts
        :param y: class_label, [n_samples]
        :return: [n_class, n_features]
        """
        if self.use_idf:
            labelbin = LabelBinarizer()
            # 计算样本属于哪个类别 [n_samples, n_classes]
            Y = labelbin.fit_transform(y)
            self.classes_ = labelbin.classes_

            # 计算类别下的文档数 [n_classes]
            class_count_ = np.sum(Y, axis=0)
            class_size = class_count_.shape[0]

            # 计算每个特征词属于每个类别的样本数 [n_classes, n_features]
            class_df_ = vectorize.class_df(X, Y)

            # 计算类别下的词汇数 [n_classes]
            self.class_freq_ = np.sum(safe_sparse_dot(Y.T, X), axis=1)

            # 计算出现特征词的类别数 [n_features]
            feature_count_ = np.sum(vectorize.tobool(class_df_), axis=0)

            # 如果特征词所在的类别不确定或不知道时，用这个特征词出现的总样本数来代替
            unknow_class_count_ = np.array([np.sum(class_count_, axis=0)])
            class_count_ = np.concatenate((class_count_, unknow_class_count_))

            unknow_class_df_ = np.sum(class_df_, axis=0).reshape(1, -1)
            class_df_ = np.concatenate((class_df_, unknow_class_df_), axis=0)

            unknow_class_freq_ = np.array([np.sum(self.class_freq_, axis=0)])
            self.class_freq_ = np.concatenate((self.class_freq_, unknow_class_freq_))

            self.classes_ = np.concatenate((self.classes_, np.array(["unknow"])), axis=0)

            # smooth class_count_, class_df_, feature_count_
            class_count_ += int(self.smooth_idf)
            class_df_ += int(self.smooth_idf)
            feature_count_ += int(self.smooth_idf)

            _, n_features = X.shape

            # [n_classes, n_features]
            first_part = np.log(np.divide(class_count_.reshape(-1, 1), class_df_)) + 1.0
            # [n_features]
            second_part = np.log(class_size / feature_count_) + 1.0
            second_part_diag = sp.spdiags(second_part, diags=0, m=n_features, n=n_features)

            self._idf = safe_sparse_dot(first_part, second_part_diag)

        return self

    def transform(self, X, y):
        """Transform a count matrix to a tf or tf-idf representation

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            X must be a matrix of term counts

        y: class_label, [n_samples]

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        n_samples, n_features = X.shape
        y = np.asarray(y)
        ny_samples = y.shape[0]

        if n_samples != ny_samples:
            raise ValueError("must have the same samples")

        res = sp.csr_matrix((1, n_features))
        tf_temp = []
        # 暂且不管样本是属于哪个类别，先对每个样本的所有类别情况计算一遍，放入 tf_temp
        for row in self.class_freq_:
            tf_temp.append(np.divide(X, row))

        # 取样本所对应类别的数据
        for i in range(n_samples):
            class_label = y[i: i + 1]
            index = np.where(self.classes_ == class_label)[0][0]

            res = sp.vstack([res, tf_temp[index][i: i + 1, :]])

        X = res[1:, :]

        if self.use_idf:
            res = sp.csr_matrix((1, n_features))
            # 暂且不管样本是属于哪个类别，先对每个样本的所有类别情况计算一遍，放入 tfidf_temp
            tfidf_temp = []
            for row in self._idf:
                idf_diag = sp.spdiags(row, diags=0, m=n_features, n=n_features)
                tfidf_temp.append(safe_sparse_dot(X, idf_diag))

            # 取样本所对应类别的数据
            for i in range(n_samples):
                class_label = y[i: i + 1]
                index = np.where(self.classes_ == class_label)[0][0]

                res = sp.vstack([res, tfidf_temp[index][i: i + 1, :]])

            X = res[1:, :]

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X
