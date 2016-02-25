# encoding: utf-8
from __future__ import division

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import _document_frequency
from sklearn.preprocessing import LabelBinarizer, normalize
from sklearn.utils.extmath import safe_sparse_dot

from com.text.feature import vectorize

__author__ = 'zql'
__date__ = '16-01-23'


class TfidfImprove(object):
    """
    改进的 Tfidf 文本向量化方法
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
        don't care the specifict value in X
        :param X: sparse matrix, [n_samples, n_features]
                  a matrix of term counts
        :param y: class_label, [n_samples]
        :return: [n_class, n_features]
        """
        if self.use_idf:
            labelbin = LabelBinarizer()
            # 计算样本属于哪个类别 [n_samples, n_classes]
            Y = labelbin.fit_transform(y)
            # LabelBinarizer 对于二分类的返回结果跟多分类的返回结果有点不一样
            # so deal with binary
            if labelbin.y_type_ == "binary":
                Y = np.hstack((1 - Y, Y))
            self.classes_ = labelbin.classes_

            # 计算每个特征词属于每个类别的样本数 [n_classes, n_features]
            class_df_ = vectorize.class_df(X, Y)

            # 如果特征词所在的类别不确定或不知道时，用这个特征词出现的总样本数来代替
            unknow_class_df_ = np.sum(class_df_, axis=0).reshape(1, -1)
            class_df_ = np.concatenate((class_df_, unknow_class_df_), axis=0)
            self.classes_ = np.concatenate((self.classes_, np.array(["unknow"])), axis=0)

            # smooth class_df_
            class_df_ += int(self.smooth_idf)

            n_samples, n_features = X.shape
            df = _document_frequency(X)

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = float(n_samples) / df
            idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)

            # [n_classes, n_features]
            self._idf = np.log(safe_sparse_dot(class_df_, idf_diag)) + 1.0

        return self

    def transform(self, X, y):
        """Transform a count matrix to a tf or tf-idf representation

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts

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

#            i = 0
#            for row in X:
#                class_label = y[i: i + 1]
#                index = np.where(self.classes_ == class_label)[0][0]
#                idf_diag = sp.spdiags(self._idf[index: index + 1, :],
#                                      diags=0, m=n_features, n=n_features)
#                res = sp.vstack([res, safe_sparse_dot(row, idf_diag)])
#
#                i += 1

            X = res[1:, :]

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

