# encoding: utf-8
from __future__ import division
from compiler.ast import flatten
import time
import math
from com.text.feature.feature import Feature
from com.text.load_sample import Load
from com.text.split_words_nlpir import SplitWords

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
    def __init__(self, f=False, subjective=True):
        super(CHIFeature, self).__init__(f, subjective)

#    def get_key_words(self, sentences):
#        # 加载训练集
#        # 每个句子还包含类别信息
#        training_datas = Load.load_training_balance()
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

    def cal_score(self, t, sentence, label, class_sentences, sentences):
        return CHIFeature.chi(t, sentence, label, class_sentences, sentences)

    @staticmethod
    def chi(t, sentence, label, all_class_datas, all_datas):
        def o_class_datas(c):
            l = []
            for c1 in all_class_datas.keys():
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
            if A == 0:
                A = 1
            if B == 0:
                B = 1
            x = N * math.pow((A * D - B * C), 2) / ((A + C) * (A + B) * (B + D) * (C + D))
            return x

#            # improved chi
#            # 特征词 t 在类别 j 各文档中出现的词频
#            t_in_df = [df.get(t, 0) for df in c_class_datas]
#            # 特征词 t 在类别 j 中的词频
#            a = sum(t_in_df)
#            # 特征词 t 词频的偏离程度
#            r = math.pow(sentence.get(t) - sum(t_in_df) / len(t_in_df), -2)
#
#            # 特征词 t 文档的偏离程度
#            b = math.pow(A - (A + B) / len(all_class_datas.keys()), 3)
#            return x * a * b * r
        N = len(all_datas)
#        chi = c_chi(label)
#        return chi
        chi = [c_chi(c_1) for c_1 in all_class_datas.keys()]
        # max
        return max(chi)

        # avg
#        prior_proba = [len(all_class_datas.get(c_1)) / N for c_1 in all_class_datas.keys()]
#        avg = 0.0
#        for i in range(len(chi)):
#            avg += chi[i] * prior_proba[i]
#        return avg

if __name__ == "__main__":
    s1 = ur"源海都学愤怒鸟的声音，好像好厉害…"
#    CHIFeature(subjective=False).get_key_words()
    CHIFeature().get_key_words()
#    IGFeature().get_key_words(s1)
