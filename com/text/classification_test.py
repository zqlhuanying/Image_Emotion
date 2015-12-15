# encoding: utf-8
import scipy.sparse as sp
from com import RESOURCE_BASE_URL
from com.text.classification import Classification
from com.text.feature.chi_feature import CHIFeature
from com.text.feature.fast_tf_idf_feature import FastTFIDFFeature
from com.text.feature.ig_feature import IGFeature
from com.text.feature.tf_idf_feature import TFIDFFeature
from com.text.load_sample import Load

__author__ = 'zql'
__date__ = '2015/12/14'


def test_classification(feature):
    # 加载数据集
    sample_url = RESOURCE_BASE_URL + "weibo_samples.xml"
    test = Load.load_test(sample_url)
    train_datas, class_label = feature.get_key_words()
    test_datas, c_true = feature.get_key_words(test)

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

# test fasttfidf
test_classification(FastTFIDFFeature())
print
print

# test tfidf
test_classification(TFIDFFeature())
print
print

# test IG
test_classification(IGFeature)
print
print

# test CHI
test_classification(CHIFeature)
print
print
