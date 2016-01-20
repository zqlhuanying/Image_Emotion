# encoding: utf-8
import scipy.sparse as sp

from com.text.classification_util import get_classification, get_objective_classification, get_emotion_classification
from com.text.feature.chi_feature import CHIFeature
from com.text.feature.fast_tf_idf_feature import FastTFIDFFeature
from com.text.feature.ig_feature import IGFeature
from com.text.feature.tf_idf_feature import TFIDFFeature
from com.text.load_sample import Load

__author__ = 'zql'
__date__ = '2015/12/14'


def test_classification(feature, incr=False):
    # 加载数据集
    if feature.subjective:
        test = Load.load_test_balance()
    else:
        test = Load.load_test_objective_balance()
    test_datas, c_true = feature.get_key_words(test)

    test = test_datas
    # 构建适合 bayes 分类的数据集
    if not sp.issparse(test_datas):
        test = feature.cal_weight(test_datas)

    clf = get_classification(feature, incr)
    c_pred_unknow = clf.predict_unknow(test)
    print c_pred_unknow
    print "precision:", clf.metrics_precision(c_true, c_pred_unknow)
    print "recall:", clf.metrics_recall(c_true, c_pred_unknow)
    print "f1:", clf.metrics_f1(c_true, c_pred_unknow)
    print "zero_one_loss:", clf.metrics_zero_one_loss(c_true, c_pred_unknow)
    print
    clf.metrics_correct(c_true, c_pred_unknow)


def classifict(feature, sentences):
    test_datas, c_true = feature.get_key_words(sentences)

    test = test_datas
    if not sp.issparse(test_datas):
        test = feature.cal_weight(test_datas)

    # 获得主客观分类器
    feature.subjective = False
    objective_clf = get_objective_classification(feature)
    # 获得情绪分类器
    feature.subjective = True
    emotion_clf = get_emotion_classification(feature, incr=False)

    c_pred = objective_clf.predict(test)
    c_pred = [emotion_clf.predict(test[i])[0] if c == "Y" else c for i, c in enumerate(c_pred)]

    print c_pred


s1 = "寂寞人生爱无休，寂寞是爱永远的主题、我和我的影子独处、它说它有悄悄话想跟我说、" \
     "它说它很想念你，原来我和我的影子，都在想你。"
classifict(CHIFeature(), s1)
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
test_classification(CHIFeature(subjective=True))
print
print
