# encoding: utf-8
import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_classification
from unbalanced_dataset import SMOTE

from com.text.feature.chi_feature import CHIFeature

__author__ = 'zql'
__date__ = '16-02-22'


def smote(X, y):
    """
    This object is an implementation of SMOTE - Synthetic Minority
    Over-sampling Technique, and the variations Borderline SMOTE 1, 2 and
    SVM-SMOTE.
    * It does not support multiple classes automatically, but can be called
    multiple times
    :return:
    """
    Y = np.asanyarray(y)
    sm = SMOTE(verbose=True)
    overx, overy = sm.fit_transform(X, Y)
    return overx, overy


if __name__ == "__main__":
    feature = CHIFeature()
    train_datas, class_label = feature.get_key_words()
    train = train_datas
    if not sp.issparse(train_datas):
        train = feature.cal_weight_improve(train_datas, class_label)

    smote(train, class_label)
    print

