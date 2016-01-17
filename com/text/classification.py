# encoding: utf-8
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import precision_score, recall_score, f1_score
from com import EMOTION_CLASS, OBJECTIVE_CLASS
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
    def __init__(self, bayes=Bayes(), subjective=True):
        self.subjective = subjective
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

    def predict_unknow(self, test_datas):
        """
        预测，若预测的概率小于一定的阖值，则输出 unknow
        :param test_datas:
        :return:
        """
        _max = 0.3

        fit_test_datas = test_datas
        if not sp.issparse(test_datas):
            fit_test_datas = self.feature_hasher.transform(test_datas)

        # proba: [n_samples, n_class]
        proba = self.bayes.predict_proba(fit_test_datas)
        max_proba_index = np.argmax(proba, axis=1)

        rowid = 0
        res = []
        for row in proba:
            index = max_proba_index[rowid]
            c = self.bayes.classes_[index] if row[index] > _max else "unknow"
            res.append(c)
            rowid += 1
        return res

    def metrics_precision(self, c_true, c_pred):
        fit_true_pred = self.__del_unknow(c_true, c_pred)
        c_true_0 = fit_true_pred[0]
        c_pred_0 = fit_true_pred[1]
        classes = self.getclasses()
        pos_label, average = self.__get_label_average(classes)
        return precision_score(c_true_0, c_pred_0, labels=classes, pos_label=pos_label, average=average)

    def metrics_recall(self, c_true, c_pred):
        fit_true_pred = self.__del_unknow(c_true, c_pred)
        c_true_0 = fit_true_pred[0]
        c_pred_0 = fit_true_pred[1]
        classes = self.getclasses()
        pos_label, average = self.__get_label_average(classes)
        return recall_score(c_true_0, c_pred_0, labels=classes, pos_label=pos_label, average=average)

    def metrics_f1(self, c_true, c_pred):
        fit_true_pred = self.__del_unknow(c_true, c_pred)
        c_true_0 = fit_true_pred[0]
        c_pred_0 = fit_true_pred[1]
        classes = self.getclasses()
        pos_label, average = self.__get_label_average(classes)
        return f1_score(c_true_0, c_pred_0, labels=classes, pos_label=pos_label, average=average)

    def metrics_correct(self, c_true, c_pred):
        # 不能过滤 unknow 部分，因统计时需要
        self._correct(c_true, c_pred, labels=self.getclasses())

    def _correct(self, c_true, c_pred, labels=None):
        def _diff(label_tuple):
            """
            if label0 == label1:
                return the (index + 1) of label0 in labels
            if label0 != label1 and label1 == 'unknow':
                return the (-1) * (index + 1) of label0 in labels
            else:
                return 0

            :param label0:
            :param label1:
            :return:
            """
            label0, label1 = label_tuple

            if label0 == label1:
                return labels.index(label0) + 1
            elif label1 == "unknow":
                return (labels.index(label0) + 1) * (-1)
            else:
                return 0

        if len(c_true) != len(c_pred):
            raise ValueError("the two lists have different size!")

        present_labels = set(c_true)
        if labels is None:
            labels = list(present_labels)

        # 每个类别共有的样本数
        true_sum = [c_true.count(c) for c in labels]

        # 每个类别预测的样本数
        pred = c_pred if isinstance(c_pred, list) else c_pred.tolist()
        pred_sum = [pred.count(c) for c in labels]

        # 每个类别下预测的正确数
        diff = map(_diff, zip(c_true, c_pred))
        tp_sum = [diff.count(i + 1) for i, c in enumerate(labels)]

        # 每个类别 unknow 数
        unknow_sum = [diff.count(-i - 1) for i, c in enumerate(labels)]

        # print
        for i, c in enumerate(labels):
            print(c)
            print("total samples: %d" % true_sum[i])
            print("predict correct: %d" % tp_sum[i])
            print("predict others into this class: %d" % (pred_sum[i] - tp_sum[i]))
            print("predict unknow: %d" % unknow_sum[i])
            print("predict incorrect: %d" % (true_sum[i] - tp_sum[i] - unknow_sum[i]))
            print

    def getclasses(self):
        if self.subjective:
            classes = EMOTION_CLASS.keys()
        else:
            classes = OBJECTIVE_CLASS.keys()
        return classes

    def __get_label_average(self, classes):
        if len(classes) <= 1:
            raise ValueError("must two classes")
        elif len(classes) <= 2:
            return "Y", "binary"
        else:
            return 1, "macro"

    @staticmethod
    def __del_unknow(c_true, c_pred):
        """
        过滤 c_pred 中 unknow 部分
        :param c_true:
        :param c_pred:
        :return:
        """
        if len(c_true) != len(c_pred):
            raise ValueError("the two lists have different size!")

        l = filter(lambda x: x[1] != "unknow", zip(c_true, c_pred))
        return zip(*l)

if __name__ == "__main__":
    print
    # 加载情绪分类数据集
    feature = CHIFeature()
    test = Load.load_test_balance()
    train_datas, class_label = feature.get_key_words()
    test_datas, c_true = feature.get_key_words(test)

    train = train_datas
    test = test_datas
    # 构建适合 bayes 分类的数据集
    if not sp.issparse(train_datas):
        train = feature.cal_weight(train_datas)
        test = feature.cal_weight(test_datas)

    clf = Classification()
    clf.get_classificator(train, class_label)
    c_pred = clf.predict(test)
    c_pred_unknow = clf.predict_unknow(test)
    print c_pred
    print "precision:", clf.metrics_precision(c_true, c_pred_unknow)
    print "recall:", clf.metrics_recall(c_true, c_pred_unknow)
    print "f1:", clf.metrics_f1(c_true, c_pred_unknow)
    print
    clf.metrics_correct(c_true, c_pred_unknow)

    # 加载主客观分类数据集
#    feature = CHIFeature(subjective=False)
#    test = Load.load_test_objective_balance()
#    train_datas, class_label = feature.get_key_words()
#    test_datas, c_true = feature.get_key_words(test)
#
#    train = train_datas
#    test = test_datas
#    # 构建适合 bayes 分类的数据集
#    if not sp.issparse(train_datas):
#        train = feature.cal_weight(train_datas)
#        test = feature.cal_weight(test_datas)
#
#    clf = Classification(subjective=False)
#    clf.get_classificator(train, class_label)
#    c_pred = clf.predict(test)
#    c_pred_unknow = clf.predict_unknow(test)
#    print c_pred
#    print "precision:", clf.metrics_precision(c_true, c_pred_unknow)
#    print "recall:", clf.metrics_recall(c_true, c_pred_unknow)
#    print "f1:", clf.metrics_f1(c_true, c_pred_unknow)
#    print
#    clf.metrics_correct(c_true, c_pred_unknow)
