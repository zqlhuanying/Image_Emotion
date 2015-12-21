# encoding: utf-8
from compiler.ast import flatten
import time
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from com import TEST_BASE_URL
from com.image.utils.common_util import CommonUtil
from com.text.feature.tf_idf_feature import TFIDFFeature
from com.text.load_sample import Load
from com.text.split_words import SplitWords

__author__ = 'zql'
__date__ = '2015/12/15'

feature_hasher = FeatureHasher(n_features=20000, non_negative=True)


def get_dict(l):
    d = {}
    for l1 in set(l):
        d[l1] = l.count(l1)
    return d


def check_train_feature():
    # train_feature
    key_words, _ = TFIDFFeature().get_key_words()
    fit_train_datas = [d.get("sentence") for d in key_words]
    train_feature = feature_hasher.transform(fit_train_datas).toarray()

    feature_count = np.sum(train_feature, axis=0)
    CommonUtil.img_array_to_file(TEST_BASE_URL + "train_feature.txt", feature_count.reshape(-1, 1))


def check_test_feature():
    test_datas = Load.load_test()
    # test feature
    key_words, _ = TFIDFFeature().get_key_words(test_datas)
    fit_test_datas = [d.get("sentence") for d in key_words]
    test_feature = feature_hasher.transform(fit_test_datas).toarray()

    feature_count = np.sum(test_feature, axis=0)
    CommonUtil.img_array_to_file(TEST_BASE_URL + "test_feature.txt", feature_count.reshape(-1, 1))
    print

check_train_feature()
check_test_feature()
