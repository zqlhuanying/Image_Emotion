# encoding: utf-8
from __future__ import division
from compiler.ast import flatten
import math
import time
from com import RESOURCE_BASE_URL, EMOTION_CLASS
from com.text.feature.feature import Feature
from com.text.load_sample import Load
from com.text.split_words import SplitWords

__author__ = 'root'
__date__ = '15-11-24'


class IGFeature(Feature):
    """
    文本的信息增益特征
    IG（T） = H（C） - H（C|T）
    """
    def __init__(self):
        super(IGFeature, self).__init__()

    def get_key_words(self, sentences):
        # 加载训练集
        sample_url = RESOURCE_BASE_URL + "weibo_samples.xml"
        # 每个句子还包含类别信息
        training_datas = Load.load_training(sample_url)
        # 去除无用信息，只留下文本信息
        pure_training_datas = [data.get("sentence") for data in training_datas]

        # 获取所有类别下的文本
        all_class_datas = IGFeature.all_class_text(training_datas)

        # 分词
        print "Before Split: ", int(time.time())
        sentence_list = list()
        sentence_list.append(sentences)
        sentence_list.append(pure_training_datas)
        SplitWords.__init__()
        splited_words_list = [SplitWords.split_words(sentence) for sentence in flatten(sentence_list)]
        SplitWords.close()
        print "After Split: ", int(time.time())

        hc = IGFeature.hc(all_class_datas, len(training_datas))
        print("hc is: %f" % hc)
        for splited_words in splited_words_list:
            print
            scores = {word: hc - IGFeature.hct(pure_training_datas, all_class_datas, word)
                      for word in set(splited_words)}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for word, score in sorted_words[:min(10, len(sorted_words))]:
                print("\tWord: %s, IG: %f" % (word.decode("utf_8"), score))

    @staticmethod
    def hct(all_datas, all_class_datas, t):
        # 计算 H（C|T）的条件熵
        # todo
        # 特征词在整个训练集或在某个类别中不出现的情况，会导致某些概率为0，怎么解决？
        # 暂定：不像 TFIDF 那样做平滑处理，而是直接排除 0 的情况
        def pt():
            # 特征T出现的概率，只要用出现过T的文档数除以总文档数
            return IGFeature.n_contains(t, all_datas) / len(all_datas)

        def pct(each_class_datas):
            # 出现T的时候，类别Ci出现的概率，只要用出现了T并且属于类别Ci的文档数除以出现了T的文档数
            return 0 if pt_num == 0 else IGFeature.n_contains(t, each_class_datas) / pt_num

        def npct(each_class_datas):
            # not in pct()
            npt_num = 1 - pt_num
            return 0 if npt_num == 0 else (len(each_class_datas) - IGFeature.n_contains(t, each_class_datas)) / npt_num

        s = 0
        pt_num = pt()
        for c in EMOTION_CLASS.keys():
            pct_num = pct(all_class_datas.get(c))
            npct_num = npct(all_class_datas.get(c))
            if pct_num != 0:
                s += pt_num * pct_num * math.log(pct_num, 2)
            if npct_num != 0:
                s += (1 - pt_num) * npct_num * math.log(npct_num, 2)

        return s * (-1.0)

    @staticmethod
    def hc(all_class_datas, datas_size):
        # 计算 H(C) 的信息熵
        s = 0
        for c in EMOTION_CLASS.keys():
            p_each_class = len(all_class_datas.get(c)) / datas_size
            if p_each_class != 0:
                s += p_each_class * math.log(p_each_class, 2)

        return s * (-1.0)

if __name__ == "__main__":
    s1 = ur"源海都学愤怒鸟的声音，好像好厉害…"
    IGFeature().get_key_words(s1)

