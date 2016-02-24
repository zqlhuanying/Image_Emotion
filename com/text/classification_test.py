# encoding: utf-8
import os

import scipy.sparse as sp
import time

from com import RESOURCE_BASE_URL
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
    test_datas, c_true = feature.get_key_words(test)

    test = test_datas
    # 构建适合 bayes 分类的数据集
    if not sp.issparse(test_datas):
        test = feature.cal_weight(test_datas)

    c_pred_unknow = clf.predict_unknow(test)
    print c_pred_unknow
    print "precision:", clf.metrics_precision(c_true, c_pred_unknow)
    print "recall:", clf.metrics_recall(c_true, c_pred_unknow)
    print "f1:", clf.metrics_f1(c_true, c_pred_unknow)
    print "zero_one_loss:", clf.metrics_zero_one_loss(c_true, c_pred_unknow)
    print
    clf.metrics_correct(c_true, c_pred_unknow)


def classifict(feature, sentences, out=False):
    if isinstance(sentences, basestring):
        sentences = [sentences]
    # 获得主客观分类器
    feature.subjective = False
    objective_clf = get_objective_classification(feature)

    # 测试集
    # 主客观部分
    test_datas, c_true = feature.get_key_words(sentences)

    test = test_datas
    if not sp.issparse(test_datas):
        test = feature.cal_weight(test_datas)

    c_pred = objective_clf.predict(test)

    # 获得情绪分类器
    feature.subjective = True
    emotion_clf = get_emotion_classification(feature, incr=False)

    # 测试集
    # 情绪部分
    test_datas, c_true = feature.get_key_words(sentences)

    test = test_datas
    if not sp.issparse(test_datas):
        test = feature.cal_weight(test_datas)

    c_pred = [emotion_clf.predict(test[i])[0] if c == "Y" else c for i, c in enumerate(c_pred)]

    if out:
        dir_ = os.path.join(RESOURCE_BASE_URL, "out")
        FileUtil.mkdirs(dir_)
        current = time.strftime('%Y-%m-%d %H:%M:%S')
        o = os.path.join(dir_, current + ".xml")

        with open(o, "w") as fp:
            [fp.write(
                    """<weibo emotion-type="%s">
    <sentence emotion-1-type="%s" emotion-2-type="none" emotion-tag="%s">
        %s
    </sentence>
</weibo>
""" % (c_pred[i], c_pred[i], "N" if c_pred[i] == "N" else "Y", s))
             for i, s in enumerate(sentences)]
    else:
        print c_pred


s1 = "寂寞人生爱无休，寂寞是爱永远的主题、我和我的影子独处、它说它有悄悄话想跟我说、" \
     "它说它很想念你，原来我和我的影子，都在想你。"
classifict(CHIFeature(), [s1,s1], out=True)
print
print


# test fasttfidf
# test_classification(FastTFIDFFeature())
print
print

# test tfidf
# test_classification(TFIDFFeature())
print
print

# test IG
# test_classification(IGFeature())
print
print

# test CHI
# test_classification(CHIFeature(subjective=True))
print
print
