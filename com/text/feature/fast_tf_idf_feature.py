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
        self.feature_hasher = FeatureHasher(n_features=60000, non_negative=True)
        super(FastTFIDFFeature, self).__init__()

    def _collect(self, splited_words_list, sentence_size):
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

if __name__ == "__main__":
    FastTFIDFFeature().get_key_words()


