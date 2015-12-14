# encoding: utf-8
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import precision_score, recall_score, f1_score
from com import RESOURCE_BASE_URL
from com.text.bayes import Bayes
from com.text.feature.chi_feature import CHIFeature
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
        self.feature_hasher = FeatureHasher(n_features=30000, non_negative=True)

    def get_classificator(self, train_datas):
        """
        获取分类器
        :return:
        """
        # 构建适合 bayes 分类的数据集
        datas = [data.get("sentence") for data in train_datas]
        class_label = [data.get("emotion-1-type") for data in train_datas]
        fit_train_datas = self.feature_hasher.transform(datas).toarray()

        # 训练模型
        self.bayes.fit(fit_train_datas, class_label)
        return self

    def predict(self, test_datas):
        datas = [data.get("sentence") for data in test_datas]
        fit_test_datas = self.feature_hasher.transform(datas).toarray()
        return self.bayes.predict(fit_test_datas)

    def metrics_precision(self, c_true, c_pred):
        return precision_score(c_true, c_pred, average="macro")

    def metrics_recall(self, c_true, c_pred):
        return recall_score(c_true, c_pred, average="macro")

    def metrics_f1(self, c_true, c_pred):
        return f1_score(c_true, c_pred, average="macro")

if __name__ == "__main__":
    # 加载数据集
    sample_url = RESOURCE_BASE_URL + "weibo_samples.xml"
    test = Load.load_test(sample_url)
    train_datas = TFIDFFeature().get_key_words()
    test_datas = TFIDFFeature().get_key_words(test)
    c_true = [data.get("emotion-1-type") for data in test_datas]

    clf = Classification()
    clf.get_classificator(train_datas)
    c_pred = clf.predict(test_datas)
    print c_pred
    print "precision:", clf.metrics_precision(c_true, c_pred)
    print "recall:", clf.metrics_recall(c_true, c_pred)
    print "f1:", clf.metrics_f1(c_true, c_pred)
