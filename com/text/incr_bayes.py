# encoding: utf-8
from __future__ import division

import os

import numpy as np
import scipy.sparse as sp

from com.constant.constant import TEXT_OUT, EMOTION_CLASS
from com.plot import plot
from com.text.bayes import Bayes
from com.text.classification import Classification
from com.text.feature.chi_feature import CHIFeature
from com.text.load_sample import Load
from com.utils.fileutil import FileUtil

__author__ = 'zql'
__date__ = '15-12-27'


class IncrBayes(Bayes):
    """
    增量式贝叶斯
    """
    def __init__(self, alpha=0.01):
        super(IncrBayes, self).__init__(alpha)

    def update(self, c_pred, sentence, copy=False):
        """
        更新类别概率和条件概率
        更新 class_count 样本数
        更新 feature_count
        :param c_pred:
        :param sentence: sparse matrix, [1, n_features]
        :param copy: False: 表示真正的更新，将修改原始的数据
                     True: 不修改原始的数据，返回一个副本
        :return:
        """
        if copy:
            class_prob_out = None
            feature_prob_out = None
            class_count_out = None
            feature_count_out = None
        else:
            class_prob_out = self.class_log_prior_
            feature_prob_out = self.feature_log_prob_
            class_count_out = self.class_count_
            feature_count_out = self.feature_count_

        self._update_class_count(c_pred, class_count_out)
        o1 = self._update_class_prob(c_pred, class_prob_out)
        # todo
        # 个人倾向于先更新 feature_count 再更新 feature_log_prob
        self._update_feature_count(c_pred, sentence, feature_count_out)
        o2 = self._update_feature_prob(c_pred, sentence, feature_prob_out)
#        self._update_feature_count(c_pred, sentence, feature_count_out)

        return o1, o2

    def _update_class_prob(self, c_pred, out=None):
        # 考虑 log 的影响
        """
        更新类别概率
        :param c_pred:
        :return:
        """
        n_classes = len(self.classes_)
        n_samples = np.sum(self.class_count_)
        # 系数
        a = (n_classes + n_samples) / (1 + n_classes + n_samples)
        # 相加项
        b = 1 / (1 + n_classes + n_samples)
        # 相加矩阵
        b_matrix = np.array([b if c == c_pred else 0 for c in self.classes_])

        o = np.exp(self.class_log_prior_, out)
        o = np.log(o * a + b_matrix, out)

        return o

    def _update_feature_prob(self, c_pred, sentence, out=None):
        index = np.where(self.classes_ == c_pred)[0][0]
        # 获得与 c_pred 相对应的那一类的数据
        correct_row = self.feature_count_[index: index + 1, :]

        # 计算 c_pred 类别下的单词数
        dci = np.sum(np.any(correct_row, axis=0))
        # 系数
        a = (1 + dci) / (2 + dci)
        # 相加项
        b = 1 / (2 + dci)
        # 相加矩阵
        # todo
        # 暂时还无法检验，暂且认为构建的 b_matrix 是正确的
#        l = []
#        d = {}
#        for k, v in sentence.items():
#            d[k] = v * b
#        l.append(d)
#        fit_sentence = Feature_Hasher.transform(l).toarray()
        fit_sentence = sentence.multiply(b).toarray()
        bool_diff = np.logical_and(correct_row, fit_sentence)
        b_matrix = np.select(bool_diff, fit_sentence)

        copy_feature_log_prob_ = self.feature_log_prob_.copy()
        prob = copy_feature_log_prob_[index: index + 1]
        np.exp(prob, prob)
        np.log(prob * a + b_matrix, prob)
        np.power(copy_feature_log_prob_, 1, out)
        return copy_feature_log_prob_

    def _update_class_count(self, c_pred, out=None):
        b_matrix = np.array([1 if c == c_pred else 0 for c in self.classes_])
        o = np.power(self.class_count_ + b_matrix, 1, out)
        return o

    def _update_feature_count(self, c_pred, sentence, out=None):
        # todo
        """
        feature_count_: 存储的是每个类别下每个特征词的权重信息
        这样一来，就不太好更新，因为权重值不能通过简单的相加来更新
        要想正确的更新 feature_count_，就必须获取到原始数据，加上 sentence 后，重新计算权重值
        但是在更新 feature_log_prob_ 时， feature_count_ 的作用有两点
        1. 根据 feature_count_ 来计算单词数（只能将非零的特征词记为 1）
        2. 根据 feature_count_ 来构建 b_matrix
        从以上的作用来看， feature_count_ 只关心是否为零，具体的数值并不关心，所以这里采用简单的
        更新方式，更新过后 feature_count_ 的数值就没什么意义了，以后不能对数值进行处理。只有在是否为零
        的情况下， feature_count_ 还是可以使用的
        :param c_pred:
        :param sentence:
        :param out:
        :return:
        """
        index = np.where(self.classes_ == c_pred)[0][0]
        # 获得与 c_pred 相对应的那一类的数据
        copy_feature_count = self.feature_count_.copy()
        correct_row = copy_feature_count[index: index + 1, :]
#        l = [sentence]
#        fit_sentence = Feature_Hasher.transform(l).toarray()
        b_matrix = sentence.toarray()
        np.power(correct_row + b_matrix, 1, correct_row)
        np.power(copy_feature_count, 1, out)
        return copy_feature_count

if __name__ == "__main__":
    # 加载情绪分类数据集
    feature = CHIFeature()
    train_datas, class_label, _ = feature.get_key_words()
    train = train_datas
    if not sp.issparse(train_datas):
        train = feature.cal_weight_improve(train_datas, class_label)

    test = Load.load_test_balance()
    test_datas, test_label, _ = feature.get_key_words(test)
    test = test_datas
    # 构建适合 bayes 分类的数据集
    if not sp.issparse(test_datas):
        test = feature.cal_weight_improve(test_datas, test_label)

    crossvalidate = False
    # 若不交叉验证 记得修改 load_sample.py 中加载 train 的比例
    if crossvalidate:
        out = os.path.join(TEXT_OUT, "best_train_test_index/test_index.txt")
        if not FileUtil.isexist(out) or FileUtil.isempty(out):
            clf0 = Classification()
            clf0.cross_validation(train, class_label, score="recall")
        test_index = np.loadtxt(out, dtype=int)
        test = train[test_index]
        test_label = np.asanyarray(class_label)[test_index].tolist()

    method_options = ("second", "four", "five")
    method_options_0 = ("B", "C", "D")
    linestyle = (':', '--', '-')
    plot.get_instance()
    for i in range(len(method_options)):
        bayes = IncrBayes()
        clf = Classification(bayes=bayes)
        clf.get_classificator(train, class_label, iscrossvalidate=crossvalidate,
                              isbalance=False, minority_target=EMOTION_CLASS.keys())
#        clf.get_classificator(train, class_label, isbalance=True, minority_target=["anger", "fear", "surprise"])
        if(i == 0):
            pred = clf.predict(test)
            pred_unknow = clf.predict_unknow(test)

            print "origin precision:", clf.metrics_precision(test_label, pred_unknow)
            print "origin recall:", clf.metrics_recall(test_label, pred_unknow)
            print "origin f1:", clf.metrics_f1(test_label, pred_unknow)
            print "origin accuracy:", clf.metrics_accuracy(test_label, pred_unknow)
            print "origin zero_one_loss:", clf.metrics_zero_one_loss(test_label, pred_unknow)
            test_proba = clf.predict_max_proba(test)
            print "origin my_zero_one_loss:", clf.metrics_my_zero_one_loss(test_proba)
            print
            clf.metrics_correct(test_label, pred_unknow)
#            plot.plot_roc(test_label, clf.predict_proba(test), classes=clf.bayes.classes_.tolist(), text='origin')

#        bayes.update(c_pred[0], test_datas[0].get("sentence"))
        incr_train_datas = Load.load_incr_datas()
        incr_train, incr_class_label, _ = feature.get_key_words(incr_train_datas)
        # 构建适合 bayes 分类的增量集
        fit_incr_train = incr_train
        if not sp.issparse(incr_train):
            fit_incr_train = feature.cal_weight_improve(incr_train, incr_class_label)

        clf.get_incr_classificator(fit_incr_train, incr_class_label, train, class_label, method=method_options[i])
        pred_unknow = clf.predict_unknow(test)

        print "incr precision:", clf.metrics_precision(test_label, pred_unknow)
        print "incr recall:", clf.metrics_recall(test_label, pred_unknow)
        print "incr f1:", clf.metrics_f1(test_label, pred_unknow)
        print "incr accuracy:", clf.metrics_accuracy(test_label, pred_unknow)
        print "incr zero_one_loss:", clf.metrics_zero_one_loss(test_label, pred_unknow)
        test_proba = clf.predict_max_proba(test)
        print "incr my_zero_one_loss:", clf.metrics_my_zero_one_loss(test_proba)
        print
        clf.metrics_correct(test_label, pred_unknow)
        plot.plot_roc(test_label, clf.predict_proba(test),
                      linestyle=linestyle[i],
                      classes=clf.bayes.classes_.tolist(),
                      text='incr ' + method_options_0[i])
    plot.show()
