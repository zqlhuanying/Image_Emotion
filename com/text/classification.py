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
        self.feature_hasher = FeatureHasher(n_features=60000, non_negative=True)

    def get_classificator(self, train_datas, class_label):
        """
        获取分类器
        :return:
        """
        fit_train_datas = train_datas
        if not sp.issparse(train_datas):
            fit_train_datas = self.feature_hasher.transform(train_datas).toarray()

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
            fit_test_datas = self.feature_hasher.transform(test_datas).toarray()

        # 预测
        return self.bayes.predict(fit_test_datas)

    def metrics_precision(self, c_true, c_pred):
        return precision_score(c_true, c_pred, labels=EMOTION_CLASS.keys(), average="macro")

    def metrics_recall(self, c_true, c_pred):
        return recall_score(c_true, c_pred, labels=EMOTION_CLASS.keys(), average="macro")

    def metrics_f1(self, c_true, c_pred):
        return f1_score(c_true, c_pred, labels=EMOTION_CLASS.keys(), average="macro")

if __name__ == "__main__":
    # 加载数据集
    test = Load.load_test_balance()
    train_datas, class_label = TFIDFFeature().get_key_words()
    test_datas, c_true = TFIDFFeature().get_key_words(test)
#    c_true = [data.get("emotion-1-type") for data in test_datas]

    train = train_datas
    test = test_datas
    # 构建适合 bayes 分类的数据集
    if not sp.issparse(train_datas):
        train = [data.get("sentence") for data in train_datas]
        test = [data.get("sentence") for data in test_datas]

    clf = Classification()
    clf.get_classificator(train, class_label)
    c_pred = clf.predict(test)
    print c_pred
    print "precision:", clf.metrics_precision(c_true, c_pred)
    print "recall:", clf.metrics_recall(c_true, c_pred)
    print "f1:", clf.metrics_f1(c_true, c_pred)
