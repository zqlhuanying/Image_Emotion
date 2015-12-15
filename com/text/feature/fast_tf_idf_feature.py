# encoding: utf-8
from compiler.ast import flatten
import time
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score
from com import RESOURCE_BASE_URL
from com.image.utils.common_util import CommonUtil
from com.text.bayes import Bayes
from com.text.feature.feature import Feature
from com.text.load_sample import Load
from com.text.split_words import SplitWords

__author__ = 'root'
__date__ = '15-12-14'


class FastTFIDFFeature(Feature):
    """
    文本的 TFIDF 特征，速度快
    """
    def __init__(self):
        # 特征 Hash 散列器
        self.feature_hasher = FeatureHasher(n_features=30000, non_negative=True)
        super(FastTFIDFFeature, self).__init__()

    def get_key_words(self, sentences=None):
        """
        获取关键词
        如果 sentences 为 None，则获取训练集中的关键词
        否则获取 sentences 中的关键词
        :return:
        """
        # 加载训练集
        sample_url = RESOURCE_BASE_URL + "weibo_samples.xml"
        # 每个句子还包含类别信息
        training_datas = Load.load_training(sample_url)

        sentence_list = training_datas
        sentence_size = len(training_datas)
        if sentences is not None:
            l = FastTFIDFFeature.__pre_process(sentences)
            sentence_list = l + training_datas
            sentence_size = len(l)

        # 分词
        print "Before Split: ", time.strftime('%Y-%m-%d %H:%M:%S')
        SplitWords.__init__()
        splited_words_list = [{"emotion-1-type": sentence.get("emotion-1-type"),
                               "sentence": SplitWords.split_words(sentence.get("sentence"))}
                              for sentence in flatten(sentence_list)]
        SplitWords.close()
        print "After Split: ", time.strftime('%Y-%m-%d %H:%M:%S')

        print "Collection datas: ", time.strftime('%Y-%m-%d %H:%M:%S')
        data = [self.get_dict(d.get("sentence")) for d in splited_words_list[: sentence_size]]
        class_label = [d.get("emotion-1-type") for d in splited_words_list[: sentence_size]]
        fit_data = self.feature_hasher.transform(data).toarray()
        tfidf = TfidfTransformer()
        tfidf.fit(fit_data)
        a = tfidf.transform(fit_data)
        print "Done: ", time.strftime('%Y-%m-%d %H:%M:%S')
        return a, class_label

    def get_dict(self, l):
        d = {}
        for l1 in set(l):
            d[l1] = l.count(l1)
        return d

    @staticmethod
    def __pre_process(sentences):
        def process_str(s):
            return {"emotion-1-type": "Unknow", "sentence": s}

        def process_dict(s):
            if not s.has_key("emotion-1-type") or not s.has_key("sentence"):
                raise AttributeError("dict has no emotion-1-type or sentence key")
            return s

        res = list()
        l = list()
        l.append(sentences)
        sentence_list = flatten(l)
        for sentence in sentence_list:
            if isinstance(sentence, basestring):
                res.append(process_str(sentence))
            if isinstance(sentence, dict):
                res.append(process_dict(sentence))
        return res

    def clf(self):
        bayes = Bayes()
        train_datas, class_label = self.get_key_words()
        bayes.fit(train_datas, class_label)

        # 加载数据集
        sample_url = RESOURCE_BASE_URL + "weibo_samples.xml"
        test = Load.load_test(sample_url)
        test_datas, c_true = self.get_key_words(test)
        c_pred = bayes.predict(test_datas)

        CommonUtil.img_array_to_file("/home/1.txt", np.array(c_true).reshape(-1, 1))
        CommonUtil.img_array_to_file("/home/2.txt", np.array(c_pred).reshape(-1, 1))
        print precision_score(c_true, c_pred)

FastTFIDFFeature().clf()
