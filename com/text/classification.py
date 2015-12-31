# encoding: utf-8
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import precision_score, recall_score, f1_score
from com import EMOTION_CLASS
from com.image.utils.common_util import CommonUtil
from com.text.bayes import Bayes
from com.text.feature.chi_feature import CHIFeature
from com.text.feature.fast_tf_idf_feature import FastTFIDFFeature
from com.text.feature.ig_feature import IGFeature
from com.text.feature.tf_idf_feature import TFIDFFeature
from com.text.load_sample import Load

__author__ = 'root'
__date__ = '15-12-13'


class Classification:
    """
    分类
    """
    def __init__(self, bayes=Bayes()):
        self.bayes = bayes
        # 特征词 Hash 散列器
        self.feature_hasher = FeatureHasher(n_features=600000, non_negative=True)

    def get_classificator(self, train_datas, class_label):
        """
        获取分类器
        :return:
        """
        fit_train_datas = train_datas
        if not sp.issparse(train_datas):
            fit_train_datas = self.feature_hasher.transform(train_datas)

        # 训练模型
        self.bayes.fit(fit_train_datas, class_label)
        return self

    def predict(self, test_datas):
        """
        预测
        :param test_datas:
        :return:
        """
        fit_test_datas = test_datas
        if not sp.issparse(test_datas):
            fit_test_datas = self.feature_hasher.transform(test_datas)

        # 预测
        return self.bayes.predict(fit_test_datas)

    def metrics_precision(self, c_true, c_pred):
        return precision_score(c_true, c_pred, labels=EMOTION_CLASS.keys(), average="macro")

    def metrics_recall(self, c_true, c_pred):
        return recall_score(c_true, c_pred, labels=EMOTION_CLASS.keys(), average="macro")

    def metrics_f1(self, c_true, c_pred):
        return f1_score(c_true, c_pred, labels=EMOTION_CLASS.keys(), average="macro")

    def metrics_correct(self, c_true, c_pred):
        self._correct(c_true, c_pred, labels=EMOTION_CLASS.keys())

    def _correct(self, c_true, c_pred, labels=None):
        if len(c_true) != len(c_pred):
            raise ValueError("the two lists have different size!")

        present_labels = set(c_true)
        if labels is None:
            labels = list(present_labels)

        # 每个类别共有的样本数
        true_sum = [c_true.count(c) for c in labels]

        # 每个类别下预测的正确数
        diff = map(lambda x: labels.index(x[0]) + 1 if x[0] == x[1] else 0, zip(c_true, c_pred))
        tp_sum = [diff.count(i + 1) for i, c in enumerate(labels)]

        # print
        for i, c in enumerate(labels):
            print(c)
            print("total samples: %d" % true_sum[i])
            print("predict correct: %d" % tp_sum[i])
            print("predict incorrect: %d" % (true_sum[i] - tp_sum[i]))
            print

if __name__ == "__main__":
    # 加载数据集
    feature = IGFeature()
    test = Load.load_test_balance()
    train_datas, class_label = feature.get_key_words()
    test_datas, c_true = feature.get_key_words(test)

    train = feature.cal_weight(train_datas)
    test = feature.cal_weight(test_datas)
    clf = Classification()
    clf.get_classificator(train, class_label)
    c_pred = clf.predict(test)
    print c_pred
    print "precision:", clf.metrics_precision(c_true, c_pred)
    print "recall:", clf.metrics_recall(c_true, c_pred)
    print "f1:", clf.metrics_f1(c_true, c_pred)
    print
    clf.metrics_correct(c_true, c_pred)
