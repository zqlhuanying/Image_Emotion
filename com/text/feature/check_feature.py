# encoding: utf-8
from compiler.ast import flatten
import numpy as np
from com import TEST_BASE_URL, RESOURCE_BASE_URL
from com.image.utils.common_util import CommonUtil
from com.text import Feature_Hasher
from com.text.feature.chi_feature import CHIFeature
from com.text.feature.ig_feature import IGFeature
from com.text.feature.tf_idf_feature import TFIDFFeature
from com.text.load_sample import Load
from com.text.utils.fileutil import FileUtil

__author__ = 'zql'
__date__ = '2015/12/15'


def get_dict(l):
    d = {}
    for l1 in set(l):
        d[l1] = l.count(l1)
    return d


def check_train_feature(feature):
    # train_feature
    key_words, _, _ = feature.get_key_words()
    fit_train_datas = [d.get("sentence") for d in key_words]
    train_feature = Feature_Hasher.transform(fit_train_datas)

    feature_count = np.sum(train_feature, axis=0)
    CommonUtil.img_array_to_file(TEST_BASE_URL + "train_feature.txt", feature_count.reshape(-1, 1))


def check_test_feature(feature):
    test_datas = Load.load_test_balance()
    # test feature
    key_words, _, _ = feature.get_key_words(test_datas)
    fit_test_datas = [d.get("sentence") for d in key_words]
    test_feature = Feature_Hasher.transform(fit_test_datas)

    feature_count = np.sum(test_feature, axis=0)
    CommonUtil.img_array_to_file(TEST_BASE_URL + "test_feature.txt", feature_count.reshape(-1, 1))
    print


def check_splited_words(feature):
    url = RESOURCE_BASE_URL + "split/" + feature.__class__.__name__ + ".txt"
    splited_size = _check_feature_size(url)
    print "Splited Words Uniquely: " + str(splited_size)


def check_key_words(feature):
    url = RESOURCE_BASE_URL + "key_words/" + feature.__class__.__name__ + ".txt"
    key_size = _check_feature_size(url)
    print "Key Words Uniquely: " + str(key_size)


def _check_feature_size(url):
    l = []
    for line in FileUtil.read(url):
        line = ",".join(line.get("sentence"))
        line = line.split(",")
        l.append(line)

    feature_size = set(flatten(l))
    return len(feature_size)

feature = CHIFeature()
check_splited_words(feature)
check_key_words(feature)
# check_train_feature(feature)
# check_test_feature(feature)
