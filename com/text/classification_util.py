# encoding: utf-8
import scipy.sparse as sp

from com.text.bayes import Bayes
from com.text.classification import Classification
from com.text.incr_bayes import IncrBayes
from com.text.load_sample import Load

__author__ = 'zql'
__date__ = '16-01-20'


def get_objective_classification(feature):
    """
    获得主客观分类器
    :param feature
    :return:
    """
    if feature.subjective:
        raise ValueError("subjective must be false")
    return get_classification(feature)


def get_emotion_classification(feature, incr=True):
    """
    获得情绪分类器
    :param feature:
    :param incr
    :return:
    """
    if not feature.subjective:
        raise ValueError("subjective must be true")
    return get_classification(feature, incr)


def get_classification(feature, incr=False):
    """
    获得分类器
    :param feature:
    :param incr
    :return:
    """
    train_datas, class_label = feature.get_key_words()

    train = train_datas
    # 构建适合 bayes 分类的数据集
    if not sp.issparse(train_datas):
        train = feature.cal_weight(train_datas)

    if incr:
        bayes = IncrBayes()
    else:
        bayes = Bayes()
    clf = Classification(bayes=bayes, subjective=feature.subjective)
    clf.get_classificator(train, class_label)
    if incr:
        # 加载测试集
        if feature.subjective:
            test = Load.load_test_balance()
        else:
            test = Load.load_test_objective_balance()

        test_datas, c_true = feature.get_key_words(test)
        test = test_datas
        # 构建适合 bayes 分类的数据集
        if not sp.issparse(test_datas):
            test = feature.cal_weight(test_datas)

        incr_train_datas = Load.load_incr_datas()
        clf.get_incr_classificator(incr_train_datas, test, c_true)
    return clf

