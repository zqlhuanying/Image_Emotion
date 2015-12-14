# encoding: utf-8
from com import RESOURCE_BASE_URL
from com.text.classification import Classification
from com.text.feature.chi_feature import CHIFeature
from com.text.feature.ig_feature import IGFeature
from com.text.feature.tf_idf_feature import TFIDFFeature
from com.text.load_sample import Load

__author__ = 'zql'
__date__ = '2015/12/14'


def test_classification(feature):
    # 加载数据集
    sample_url = RESOURCE_BASE_URL + "weibo_samples.xml"
    test = Load.load_test(sample_url)
    train_datas = feature.get_key_words()
    test_datas = feature.get_key_words(test)
    c_true = [data.get("emotion-1-type") for data in test_datas]

    clf = Classification()
    clf.get_classificator(train_datas)
    c_pred = clf.predict(test_datas)
    print c_pred
    print "precision:", clf.metrics_precision(c_true, c_pred)
    print "recall:", clf.metrics_recall(c_true, c_pred)

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
