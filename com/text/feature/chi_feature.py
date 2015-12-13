# encoding: utf-8
from __future__ import division
from compiler.ast import flatten
import time
import math
from com import EMOTION_CLASS, RESOURCE_BASE_URL
from com.text.feature.feature import Feature
from com.text.load_sample import Load
from com.text.split_words import SplitWords

__author__ = 'zql'
__date__ = '2015/11/27'


class CHIFeature(Feature):
    """
    文本的开方检验特征

    CHI = N * (A * D - B * c) / [(A + C) * (A + B) * (B + D) * (C + D)]

    N : 总文档数
    M : 某一类别下的总文档数
                        属于类别 C          不属于类别 C
    包含特征词 T             A                   B
    不包含特征词 T           C                   D
    """
    def __init__(self):
        super(CHIFeature, self).__init__()

#    def get_key_words(self, sentences):
#        # 加载训练集
#        sample_url = RESOURCE_BASE_URL + "weibo_samples.xml"
#        # 每个句子还包含类别信息
#        training_datas = Load.load_training(sample_url)
#        # 去除无用信息，只留下文本信息
#        pure_training_datas = [data.get("sentence") for data in training_datas]
#
#        # 获取所有类别下的文本
#        all_class_datas = CHIFeature.all_class_text(training_datas)
#
#        # 分词
#        print int(time.time())
#        sentence_list = list()
#        sentence_list.append(sentences)
#        sentence_list.append(pure_training_datas)
#        SplitWords.__init__()
#        splited_words_list = [SplitWords.split_words(sentence) for sentence in flatten(sentence_list)]
#        SplitWords.close()
#        print int(time.time())

    def cal_weight(self, t, sentence, class_sentences, sentences):
        return CHIFeature.chi(t, class_sentences, sentences)

    @staticmethod
    def chi(t, all_class_datas, all_datas):
        def o_class_datas(c):
            l = []
            for c1 in EMOTION_CLASS.keys():
                if c1 != c:
                    l += all_class_datas.get(c1)
            return l

        def c_chi(c):
            """
            计算特征词 t 在 c 类别下的开方值
            :param c:
            :return:
            """
            c_class_datas = all_class_datas.get(c)
            other_class_datas = o_class_datas(c)
            M = len(c_class_datas)
            A = CHIFeature.n_contains(t, c_class_datas)
            B = CHIFeature.n_contains(t, other_class_datas)
            C = M - A
            D = N - A - B - C
            x = N * math.pow((A * D - B * C), 2) / ((A + C) * (A + B) * (B + D) * (C + D))
            return x
        N = len(all_datas)
        chi = [c_chi(c_1) for c_1 in EMOTION_CLASS.keys()]
        return max(chi)

if __name__ == "__main__":
    s1 = ur"源海都学愤怒鸟的声音，好像好厉害…"
    CHIFeature().get_key_words()
#    IGFeature().get_key_words(s1)