# encoding: utf-8
from __future__ import division
from compiler.ast import flatten
import time
from com import EMOTION_CLASS, RESOURCE_BASE_URL
from com.text.load_sample import Load
from com.text.split_words import SplitWords

__author__ = 'zql'
__date__ = '2015/11/12'


class Feature(object):
    """
    文本特征词抽取
    """
    def __init__(self):
        pass

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
            l = Feature.__pre_process(sentences)
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

        # 获取所有类别下的文本
        all_class_datas = Feature.all_class_text(splited_words_list)

        # return
        print "Collecting datas: ", time.strftime('%Y-%m-%d %H:%M:%S')
        res = []
        for splited_words_dict in splited_words_list[0: sentence_size]:
            splited_words = splited_words_dict.get("sentence")
            scores = {splited_word: self.cal_weight(splited_word, splited_words, all_class_datas,
                                                    [d.get("sentence") for d in splited_words_list])
                      for splited_word in set(splited_words)}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            res.append({"sentence": dict(sorted_words[:min(10, len(sorted_words))]),
                        "emotion-1-type": splited_words_dict.get("emotion-1-type")})
        print "Done: ", time.strftime('%Y-%m-%d %H:%M:%S')
        return res

        # print
#        for splited_words_dict in splited_words_list[0: sentence_size]:
#            print
#            splited_words = splited_words_dict.get("sentence")
#            scores = {splited_word: self.cal_weight(splited_word, splited_words, all_class_datas,
#                                                    [d.get("sentence") for d in splited_words_list])
#                      for splited_word in set(splited_words)}
#            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#            for word, score in sorted_words[:min(10, len(sorted_words))]:
#                print("\tWord: %s, Weight: %f" % (word.decode("utf_8"), score))

    def cal_weight(self, t, sentence, class_sentences, sentences):
        """
        计算特征词 t 的权重
        :param t: 特征词
        :param sentence: 特征词所在的句子，分词后
        :param class_sentences: 带有类别信息的句子集，即类别 c 下的句子集，最好也是分词后（不分词貌似也不影响）
        :param sentences: 句子集，最后也是分词后（不分词貌似也不影响）
        :return:
        """
        pass

    @staticmethod
    def tf(word, words):
        """
        计算词频
        :param word:
        :param words: is a list
        :return:
        """
        return words.count(word) / len(words)

    @staticmethod
    def n_contains(word, wordslist):
        # 计算特征词 word 在文档集 wordslist 中出现的文档数
        return sum(1 for words in wordslist if word in words)

    @staticmethod
    def all_class_text(datas):
        # 将 datas 下的数据以 {类别 ：[文档]} 的形式返回
        return {c: Feature.__each_class_text(datas, c) for c in EMOTION_CLASS.keys()}

    @staticmethod
    def __each_class_text(datas, c):
        # 获取 datas 下，类别 c 的文本
        if c not in EMOTION_CLASS.keys():
            raise ValueError("have no emotion class")
        return [data.get("sentence") for data in datas if data.get("emotion-1-type") == c]

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


if __name__ == "__main__":
    ss = "我在高楼高楼上高楼"
    s = "高楼"
    print ss.count(s)
    print len(ss)
    print Feature.tf(s, ss)
    if s in ss:
        print "Yes"
