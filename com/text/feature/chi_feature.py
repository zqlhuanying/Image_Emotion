# encoding: utf-8
from compiler.ast import flatten
import time
from com import EMOTION_CLASS, RESOURCE_BASE_URL
from com.text.feature.feature import Feature
from com.text.load_sample import Load
from com.text.split_words import SplitWords

__author__ = 'zql'
__date__ = '2015/11/27'


class CHIFeature(Feature):
    # todo
    """
    文本的开方检验特征

    CHI = N * (A * D - B * c) / [(A + C) * (A + B) * (B + D) * (C + D)]

    N : 总文档数
                        属于类别 C          不属于类别 C
    包含特征词 T             A                   B
    不包含特征词 T           C                   D
    """
    def __init__(self):
        super(CHIFeature, self).__init__()

    def get_key_words(self, sentences):
        # 加载训练集
        sample_url = RESOURCE_BASE_URL + "weibo_samples.xml"
        # 每个句子还包含类别信息
        training_datas = Load.load_training(sample_url)
        # 去除无用信息，只留下文本信息
        pure_training_datas = [data.get("sentence") for data in training_datas]

        # 获取所有类别下的文本
        all_class_datas = CHIFeature.all_class_text(training_datas)

        # 分词
        print int(time.time())
        sentence_list = list()
        sentence_list.append(sentences)
        sentence_list.append(pure_training_datas)
        SplitWords.__init__()
        splited_words_list = [SplitWords.split_words(sentence) for sentence in flatten(sentence_list)]
        SplitWords.close()
        print int(time.time())

    @staticmethod
    def chi(all_datas, all_class_datas, t):
        pass

    @staticmethod
    def all_class_text(datas):
        return {c: CHIFeature.__each_class_text(datas, c) for c in EMOTION_CLASS.keys()}

    @staticmethod
    def __each_class_text(datas, c):
        # 获取 datas 下，类别 c 的文本
        if c not in EMOTION_CLASS.keys():
            raise ValueError("have no emotion class")
        return [data.get("sentence") for data in datas if data.get("emotion-1-type") == c]

    @staticmethod
    def __n_contains(word, wordslist):
        return sum(1 for words in wordslist if word in words)

if __name__ == "__main__":
    s1 = ur"源海都学愤怒鸟的声音，好像好厉害…"
    CHIFeature().get_key_words(s1)
