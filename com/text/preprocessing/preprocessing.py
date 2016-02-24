# encoding: utf-8
import numpy as np
import scipy.sparse as sp
from sklearn.datasets import make_classification
from sklearn.utils import check_X_y
from unbalanced_dataset import SMOTE

from com.text.feature.chi_feature import CHIFeature
from costcla.datasets import load_creditscoring1
from costcla.sampling import smote

from com.text.preprocessing import _smote

__author__ = 'zql'
__date__ = '16-02-22'


def my_smote(X, y, minority_target=None, per=0.5):
    """
    This object is an implementation of SMOTE - Synthetic Minority
    Over-sampling Technique, and the variations Borderline SMOTE 1, 2 and
    SVM-SMOTE.
    :param X: nd-array, sparse matrix, shape=[n_samples, n_features]
    :param y: nd-array, list, shape=[n_samples]
    :param minority_target: list
    :param per
    :return:
    """
    X, Y = check_X_y(X, y, 'csr')
    unique_label = list(set(Y))
    label_count = [np.sum(Y == i) for i in unique_label]

    if minority_target is None:
        minority_index = [np.argmin(label_count)]
    else:
        minority_index = [unique_label.index(target) for target in minority_target]

    majority = np.max(label_count)
    for i in minority_index:
        N = (int((majority * 1.0 / (1 - per) - majority) / label_count[i]) - 1) * 100
        safe, synthetic, danger = _smote._borderlineSMOTE(X, Y, unique_label[i], N, k=5)
        syn_label = np.array([unique_label[i]] * synthetic.shape[0])
        X = sp.vstack([X, synthetic])
        Y = np.concatenate([Y, syn_label])

    return X, Y

if __name__ == "__main__":
    feature = CHIFeature()
    train_datas, class_label = feature.get_key_words()
    train = train_datas
    if not sp.issparse(train_datas):
        train = feature.cal_weight_improve(train_datas, class_label)

    data_smote, target_smote = my_smote(train, class_label, minority_target=["fear", "surprise"])

    print np.sum([target_smote == "fear"])
    print np.sum([target_smote == "surprise"])

#    data = load_creditscoring1()
#    print len((data.target == 0).nonzero()[0])
#    print len((data.target == 1).nonzero()[0])
#    data_smote, target_smote = smote(data.data, data.target, per=0.7)
#    print len((target_smote == 0).nonzero()[0])
#    print len((target_smote == 1).nonzero()[0])
#    print

