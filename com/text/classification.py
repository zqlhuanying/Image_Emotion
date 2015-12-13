# encoding: utf-8
from sklearn.feature_extraction import FeatureHasher
from com import RESOURCE_BASE_URL
from com.text.bayes import Bayes
from com.text.feature.tf_idf_feature import TFIDFFeature
from com.text.load_sample import Load

__author__ = 'root'
__date__ = '15-12-13'


class Classification:
    """
    分类
    """
    def __init__(self, feature, bayes=Bayes()):
        self.feature = feature
        self.bayes = bayes
        # 特征词 Hash 散列器
        self.feature_hasher = FeatureHasher(n_features=30000, non_negative=True)

    def get_classificator(self):
        """
        获取分类器
        :return:
        """
        # 提取训练集中的特征词
        train_datas = self.feature.get_key_words()

        # 构建适合 bayes 分类的数据集
        datas = [data.get("sentence") for data in train_datas]
        class_label = [data.get("emotion-1-type") for data in train_datas]
        fit_train_datas = self.feature_hasher.transform(datas).toarray()

        # 训练模型
        self.bayes.fit(fit_train_datas, class_label)
        return self

    def predict(self, test):
        test_datas = self.feature.get_key_words(test)
        datas = [data.get("sentence") for data in test_datas]
        fit_test_datas = self.feature_hasher.transform(datas).toarray()
        return self.bayes.predict(fit_test_datas)


if __name__ == "__main__":
    # 加载测试集
    sample_url = RESOURCE_BASE_URL + "weibo_samples.xml"
    test_datas = Load.load_test(sample_url)

    clf = Classification(TFIDFFeature())
    clf.get_classificator()
    c_pred = clf.predict(test_datas)
    print c_pred
