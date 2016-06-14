# encoding: utf-8
import os
from compiler.ast import flatten

from com.constant.constant import TEXT_OUT
from com.text.feature.chi_feature import CHIFeature
from com.utils.fileutil import FileUtil

__author__ = 'zql'
__date__ = '2015/12/15'


def check_splited_words(feature):
    url = os.path.join(TEXT_OUT, "split/" + feature.__class__.__name__ + ".txt")
    splited_size = _check_feature_size(url)
    print "Splited Words Uniquely: " + str(splited_size)


def check_key_words(feature):
    url = os.path.join(TEXT_OUT, "key_words/" + feature.__class__.__name__ + ".txt")
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
