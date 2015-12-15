# encoding: utf-8
from compiler.ast import flatten
import time
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from com import RESOURCE_BASE_URL
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
    sample_url = RESOURCE_BASE_URL + "weibo_samples.xml"
    train_datas = Load.load_training(sample_url)

    # train_feature
    sentence_list = train_datas
    key_words = TFIDFFeature().get_key_words(sentence_list)
    fit_train_datas = [d.get("sentence") for d in key_words]
    # 分词
#    print "Before Split: ", time.strftime('%Y-%m-%d %H:%M:%S')
#    SplitWords.__init__()
#    splited_words_list = [{"emotion-1-type": sentence.get("emotion-1-type"),
#                           "sentence": SplitWords.split_words(sentence.get("sentence"))}
#                          for sentence in flatten(sentence_list)]
#    SplitWords.close()
#    print "After Split: ", time.strftime('%Y-%m-%d %H:%M:%S')
#
#    fit_train_datas = [get_dict(d.get("sentence")) for d in splited_words_list]
    train_feature = feature_hasher.transform(fit_train_datas).toarray()

    feature_count = np.sum(train_feature, axis=0)
    CommonUtil.img_array_to_file("F://3.txt", feature_count.reshape(-1, 1))


def check_test_feature():
    sample_url = RESOURCE_BASE_URL + "weibo_samples.xml"
    test_datas = Load.load_test(sample_url)
    # test feature
    sentence_list = test_datas
    key_words = TFIDFFeature().get_key_words(sentence_list)
    fit_test_datas = [d.get("sentence") for d in key_words]
    # 分词
#    print "Before Split: ", time.strftime('%Y-%m-%d %H:%M:%S')
#    SplitWords.__init__()
#    splited_words_list = [{"emotion-1-type": sentence.get("emotion-1-type"),
#                           "sentence": SplitWords.split_words(sentence.get("sentence"))}
#                          for sentence in flatten(sentence_list)]
#    SplitWords.close()
#    print "After Split: ", time.strftime('%Y-%m-%d %H:%M:%S')

#    fit_all_datas = [get_dict(d.get("sentence")) for d in splited_words_list]
    test_feature = feature_hasher.transform(fit_test_datas).toarray()

    feature_count = np.sum(test_feature, axis=0)
    CommonUtil.img_array_to_file("F://4.txt", feature_count.reshape(-1, 1))
    print

check_train_feature()
check_test_feature()
