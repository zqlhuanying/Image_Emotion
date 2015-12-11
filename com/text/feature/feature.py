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

    def get_key_words(self, sentences):
        """
        获取关键词
        :return:
        """
        # 加载训练集
        sample_url = RESOURCE_BASE_URL + "weibo_samples.xml"
        # 每个句子还包含类别信息
        training_datas = Load.load_training(sample_url)
        # 去除无用信息，只留下文本信息
        pure_training_datas = [data.get("sentence") for data in training_datas]

        # 获取所有类别下的文本
        all_class_datas = Feature.all_class_text(training_datas)

        # 分词
        print "Before Split: ", time.strftime('%Y-%m-%d %H:%M:%S')
        sentence_list = list()
        sentence_list.append(sentences)
        sentence_list.append(pure_training_datas)
        SplitWords.__init__()
        splited_words_list = [SplitWords.split_words(sentence) for sentence in flatten(sentence_list)]
        SplitWords.close()
        print "After Split: ", time.strftime('%Y-%m-%d %H:%M:%S')

        for splited_words in splited_words_list:
            print
            scores = {splited_word: self.cal_weight(splited_word, splited_words, all_class_datas, pure_training_datas)
                      for splited_word in set(splited_words)}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for word, score in sorted_words[:min(10, len(sorted_words))]:
                print("\tWord: %s, TF-IDF: %f" % (word.decode("utf_8"), score))

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
        pass


if __name__ == "__main__":
    ss = "我在高楼高楼上高楼"
    s = "高楼"
    print ss.count(s)
    print len(ss)
    print Feature.tf(s, ss)
    if s in ss:
        print "Yes"
