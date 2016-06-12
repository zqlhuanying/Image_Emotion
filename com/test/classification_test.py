# encoding: utf-8
import os

import numpy as np
import scipy.sparse as sp
import time

from com.constant.constant import OUT_BASE_URL
from com.text import collect
from com.text.classification_util import get_classification, get_objective_classification, get_emotion_classification
from com.text.feature.chi_feature import CHIFeature
from com.text.feature.fast_tf_idf_feature import FastTFIDFFeature
from com.text.feature.ig_feature import IGFeature
from com.text.feature.tf_idf_feature import TFIDFFeature
from com.text.load_sample import Load
from com.text.utils.fileutil import FileUtil

__author__ = 'zql'
__date__ = '2015/12/14'


def test_classification(feature, incr=False):
    clf = get_classification(feature, incr)

    # 加载测试数据集
    if feature.subjective:
        test = Load.load_test_balance()
    else:
        test = Load.load_test_objective_balance()
    test_datas, c_true, _ = feature.get_key_words(test)

    test = test_datas
    # 构建适合 bayes 分类的数据集
    if not sp.issparse(test_datas):
        test = feature.cal_weight_improve(test_datas, c_true)

    c_pred_unknow = clf.predict_unknow(test)
    print c_pred_unknow
    print "precision:", clf.metrics_precision(c_true, c_pred_unknow)
    print "recall:", clf.metrics_recall(c_true, c_pred_unknow)
    print "f1:", clf.metrics_f1(c_true, c_pred_unknow)
    print "origin accuracy:", clf.metrics_accuracy(c_true, c_pred_unknow)
    print "zero_one_loss:", clf.metrics_zero_one_loss(c_true, c_pred_unknow)
    test_proba = clf.predict_max_proba(test)
    print "my_zero_one_loss:", clf.metrics_my_zero_one_loss(test_proba)
    print
    clf.metrics_correct(c_true, c_pred_unknow)


def classifict(feature, sentences, incr=False, out=False):
    if isinstance(sentences, basestring):
        sentences = [sentences]

    # 获得主客观分类器
    feature.subjective = False
    objective_clf = get_objective_classification(feature)

    # 测试集
    # 主客观部分
    test_datas_objective, c_true_objective, danger_index_objective = feature.get_key_words(sentences)

    test_objective = test_datas_objective
    if not sp.issparse(test_datas_objective):
        test_objective = feature.cal_weight_improve(test_datas_objective, c_true_objective)

    c_pred_objective = objective_clf.predict(test_objective)

    # 获得情绪分类器
    feature.subjective = True
    emotion_clf = get_emotion_classification(feature, incr=incr)

    # 测试集
    # 情绪部分
    test_datas, c_true, danger_index = feature.get_key_words(sentences)

    test = test_datas
    if not sp.issparse(test_datas):
        test = feature.cal_weight_improve(test_datas, c_true)

    c_pred = []
    for i in range(len(sentences)):
        if i not in danger_index_objective and i not in danger_index:
            before_i_in_danger_obj = np.sum(np.asarray(danger_index_objective) < i)
            before_i_in_danger_ = np.sum(np.asarray(danger_index) < i)

            c = emotion_clf.predict(test[i - before_i_in_danger_])[0] if c_pred_objective[i - before_i_in_danger_obj] == "Y"\
                else c_pred_objective[i - before_i_in_danger_obj]
            c_pred.append(c)

    if out:
        dir_ = os.path.join(OUT_BASE_URL, "out0")
        FileUtil.mkdirs(dir_)
        current = time.strftime('%Y-%m-%d %H:%M:%S')
        o = os.path.join(dir_, current + ".xml")

        with open(o, "w") as fp:
            for i, s in enumerate(sentences):
                if i not in danger_index_objective and i not in danger_index:
                    before_i_in_danger_obj = np.sum(np.asarray(danger_index_objective) < i)
                    before_i_in_danger_ = np.sum(np.asarray(danger_index) < i)
                    fp.write(
                        """<weibo emotion-type="%s">
    <sentence emotion-1-type="%s" emotion-2-type="none" emotion-tag="%s">
        %s
    </sentence>
</weibo>
""" % (c_pred[i - before_i_in_danger_], c_pred[i - before_i_in_danger_], "N" if c_pred_objective[i - before_i_in_danger_obj] == "N" else "Y", s))
                else:
                    fp.write(
                        """<weibo emotion-type="%s">
    <sentence emotion-1-type="%s" emotion-2-type="none" emotion-tag="%s">
        %s
    </sentence>
</weibo>
""" % ("None", "None", "N", s + "\n Can't recognize because it has insufficient key_words"))

    else:
        print c_pred

if __name__ == "__main__":
    if False:
        [collect.collect_weibo() for i in range(10)]

    if True:
        feature = CHIFeature()
        path = "collect"
        sentences = collect.read_weibo(path)
        sentences = [s.get("sentence") for s in sentences]
        classifict(feature, sentences, incr=True, out=True)

#        test_classification(feature, incr=True)

    if False:
        test_classification(CHIFeature(subjective=False))

#    s1 = "寂寞人生爱无休，寂寞是爱永远的主题、我和我的影子独处、它说它有悄悄话想跟我说、" \
#         "它说它很想念你，原来我和我的影子，都在想你。"
#    classifict(CHIFeature(), [s1,s1], out=True)
#    print
#    print
#
#    # test fasttfidf
#    test_classification(FastTFIDFFeature())
#    print
#    print
#
#    # test tfidf
#    test_classification(TFIDFFeature())
#    print
#    print
#
#    # test IG
#    test_classification(IGFeature())
#    print
#    print
#
#    # test CHI
#    test_classification(CHIFeature(subjective=True))
#    print
#    print
