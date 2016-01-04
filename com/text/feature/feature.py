# encoding: utf-8
from __future__ import division
from compiler.ast import flatten
import copy
import time
import math
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer

from com import EMOTION_CLASS, RESOURCE_BASE_URL, TEST_BASE_URL
from com.text.utils.fileutil import FileUtil
from com.text.load_sample import Load
from com.text.split_words_jieba import SplitWords

__author__ = 'zql'
__date__ = '2015/11/12'


class Feature(object):
    """
    文本特征词抽取
    """
    def __init__(self):
        # f 开关，将分词后的结果写入到文本中
        #   若资源有更新，可以打开开关，强制写入新的分词后的结果
        self.f = False
        self.istrain = False
        self.feature_hasher = FeatureHasher(n_features=600000, non_negative=True)

    def get_key_words(self, sentences=None):
        """
        获取关键词
        如果 sentences 为 None，则获取训练集中的关键词
        否则获取 sentences 中的关键词
        :return:
        """
        if sentences is None:
            self.istrain = True
        else:
            self.istrain = False

        splited_words_list, sentence_size = self._split(sentences)

        return self._collect(splited_words_list, sentence_size)

    def cal_weight(self, key_words):
        """
        计算获取特征词后的权重信息
        :param key_words: [{'sentence': {}}, ...] or [{}, ...]
        :return:
        """
        has_sentence = "sentence" in key_words[0]
        if has_sentence:
            key_words = [d.get("sentence") for d in key_words]
        fit_data = self.feature_hasher.transform(key_words)
        tfidf = TfidfTransformer()
        tfidf.fit(fit_data)
        weight_matrix = tfidf.transform(fit_data)
        return weight_matrix

    def cal_score(self, t, sentence, class_sentences, sentences):
        """
        计算特征词 t 的得分
        :param t: 特征词
        :param sentence: 特征词所在的句子，分词后
        :param class_sentences: 带有类别信息的句子集，即类别 c 下的句子集，最好也是分词后（不分词貌似也不影响）
        :param sentences: 句子集，最好也是分词后（不分词貌似也不影响）
        :return:
        """
        pass

    def _split(self, sentences):
        # 加载分词后的训练集
        print "Before Split: ", time.strftime('%Y-%m-%d %H:%M:%S')
        splited_words_list = self._get_splited_train()
        sentence_size = len(splited_words_list)

        if sentences is not None:
            l = Feature.__pre_process(sentences)

            splited_sentence_list = Feature.__split(flatten(l))
            # splited_sentence_list = Feature.__del_low_df_word(splited_sentence_list)
            splited_words_list = splited_sentence_list + splited_words_list
            sentence_size = len(splited_sentence_list)

        print "Split Done: ", time.strftime('%Y-%m-%d %H:%M:%S')

        return splited_words_list, sentence_size

    def _collect(self, splited_words_list, sentence_size):
        key_words_txt = RESOURCE_BASE_URL + "key_words/" + self.__class__.__name__ + ".txt"
#        def norm(word_scores):
#            """
#            以样本为单位正则化
#            归一化（正则化）
#            Normalization 主要思想是对每个样本计算其p-范数，然后对该样本中每个元素除以该范数，
#            这样处理的结果是使得每个处理后样本的p-范数（l1-norm,l2-norm）等于1。
#
#            p-范数的计算公式：||X||p=(|x1|^p+|x2|^p+...+|xn|^p)^1/p
#
#            该方法主要应用于文本分类和聚类中。
#
#            :param word_scores: a dict {word: score}
#            """
#            p = 0.0
#            for v in word_scores.values():
#                p += math.pow(math.fabs(v), 2)
#            p = math.pow(p, 1.0 / 2)
#
#            for k, v in word_scores.items():
#                word_scores[k] = v / p

#        def reduce_dim(word_scores):
#            """
#            降维：选取累加权重信息占比超过 0.9 的特征词
#            """
#            _size = len(word_scores)
#            _max = math.pow(_size, 1.0 / 2) * 0.85
#            res = {}
#            # 降序排序
#            sort = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
#            _sum = 0.0
#            for k, v in sort:
#                if(_sum > _max):
#                    break
#                res[k] = v
#                _sum += v
#            return res

        print "Cal Scores: ", time.strftime('%Y-%m-%d %H:%M:%S')
        if not self.istrain or self.f or not FileUtil.isexist(key_words_txt) or FileUtil.isempty(key_words_txt):
            if len(splited_words_list) == sentence_size:
                train_range = slice(sentence_size)
            else:
                train_range = slice(sentence_size, len(splited_words_list))

            # 获取所有类别下的文本
            all_class_datas = Feature.all_class_text(splited_words_list[train_range])

            # 获取类别标签
            class_label = [d.get("emotion-1-type") for d in splited_words_list[: sentence_size]]

            # return term/frequency or term/score
            res = []
            for splited_words_dict in splited_words_list[0: sentence_size]:
                splited_words = splited_words_dict.get("sentence")
                # 计算每个单词的得分 scores: {word: [score, frequency], ...}
                scores = {splited_word: [self.cal_score(splited_word, splited_words, all_class_datas,
                                                        [d.get("sentence") for d in splited_words_list[train_range]]),
                                         splited_words.count(splited_word)]
                          for splited_word in set(splited_words)}
                # 归一化
                # norm(scores)
                # 降维处理
                sorted_words = scores
#                if not self.istrain:
#                    sorted_words = reduce_dim(scores)

                # Collection
                # if False return term/score
                # if True  return term/frequency
#                if False:
#                    for k in sorted_words.keys():
#                        sorted_words[k] = splited_words.count(k)

                res.append({"sentence": sorted_words,
                            "emotion-1-type": splited_words_dict.get("emotion-1-type")})
            print "Cal Scores Done: ", time.strftime('%Y-%m-%d %H:%M:%S')
            # FileUtil.write(TEST_BASE_URL + "scores.txt", res)
            print "Begin Normalization: ", time.strftime('%Y-%m-%d %H:%M:%S')
            # 归一化
            self.norm(res)
            # FileUtil.write(TEST_BASE_URL + "norm.txt", res)
            print "Normalization Done: ", time.strftime('%Y-%m-%d %H:%M:%S')

            print "Begin Reduce: ", time.strftime('%Y-%m-%d %H:%M:%S')
            # 降维
            self.reduce_dim(res)
            print "Reduce Done: ", time.strftime('%Y-%m-%d %H:%M:%S')

            # Try Convert term/score to term/frequency
            # if False return term/score
            # if True  return term/frequency
            for d in res:
                ws = d.get("sentence")
                for k, v in ws.items():
                    ws[k] = v[0]
                    if True:
                        ws[k] = v[1]

            # 写入文件
            if self.istrain:
                FileUtil.write(key_words_txt, res)
            else:
                FileUtil.write(TEST_BASE_URL + "reduce.txt", res)
        else:
            res = FileUtil.read(key_words_txt)
            class_label = [r["emotion-1-type"] for r in res]

        # 输出统计信息
        if True:
            self.__print_top_key_word(res)
        return res, class_label

    def _get_splited_train(self):
        """
        优先从文件中读取训练集分词后的结果
        :return:
        """
        split_txt = RESOURCE_BASE_URL + "split/" + self.__class__.__name__ + ".txt"
        if self.f or not FileUtil.isexist(split_txt) or FileUtil.isempty(split_txt):
            # 加载训练集
            # 每个句子还包含类别信息
            training_datas = Load.load_training_balance()

            splited_words_list = Feature.__split(flatten(training_datas))
            # splited_words_list = Feature.__del_low_df_word(splited_words_list)

            FileUtil.write(split_txt, splited_words_list)
        else:
            splited_words_list = FileUtil.read(split_txt)

        return splited_words_list

    @staticmethod
    def norm(word_scores):
        """
        以类别为单位正则化
        归一化（正则化）
        Normalization 主要思想是对每个样本计算其p-范数，然后对该样本中每个元素除以该范数，
        这样处理的结果是使得每个处理后样本的p-范数（l1-norm,l2-norm）等于1。

        p-范数的计算公式：||X||p=(|x1|^p+|x2|^p+...+|xn|^p)^1/p

        该方法主要应用于文本分类和聚类中。

        :param word_scores: a list [{emotion-1-type: "like", sentence: {word: score}}, ...]
        """
        def norm_0(c0):
            word_score = reduce(Feature.union, all_class.get(c))
            p = 0.0
            for k, v in word_score.items():
                p += math.pow(math.fabs(v[0]), 2)
            p = math.pow(p, 1.0 / 2)

            for d in word_scores:
                if d.get("emotion-1-type") == c0:
                    ws = d.get("sentence")
                    for k, v in ws.items():
                        ws[k][0] = v[0] / p

        all_class = Feature.all_class_text(word_scores)
        for c in EMOTION_CLASS.keys():
            norm_0(c)

    @staticmethod
    def reduce_dim(word_scores):
        """
        降维：选取累加权重信息占比超过 0.9 的特征词
        """
        def reduce_dim_0(c0):
            word_score = reduce(Feature.union, all_class.get(c0))
            _size = len(word_score)
            _max = math.pow(_size, 1.0 / 2) * 0.85

            res = []
            # 降序排序
            sort = sorted(word_score.items(), key=lambda x: x[1][0], reverse=True)
            _sum = 0.0
            for k, v in sort:
                if(_sum > _max):
                    break
                res.append(k)
                _sum += v[0]

            for d in word_scores:
                if d.get("emotion-1-type") == c0:
                    ws = d.get("sentence")
                    for k, v in ws.items():
                        if k in res:
                            ws[k][2] = 1

        # 初始化 word_scores, 为每个单词增加一个标记 mark
        # 初始值 0, 若这个单词允许保留，则改为 1
        for d in word_scores:
            ws = d.get("sentence")
            for k, v in ws.items():
                ws[k].append(0)

        all_class = Feature.all_class_text(word_scores)

        [reduce_dim_0(c) for c in EMOTION_CLASS.keys()]

        # 保持 word_scores 的形式不变，删除 mark 标记
        # 若 1, 则只需要删除 mark 标记；若 0，则删除单词
        for d in word_scores:
            ws = d.get("sentence")
            ws_0 = copy.deepcopy(ws)
            for k, v in ws_0.items():
                if v[2] == 1:
                    ws[k].pop()
                else:
                    del ws[k]

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

    @staticmethod
    def __split(sentence_list):
        SplitWords.__init__()

        l = []
        for sentence in sentence_list:
            splited_words = SplitWords.split_words(sentence.get("sentence"))
            if splited_words:
                d = {}
                d["emotion-1-type"] = sentence.get("emotion-1-type")
                d["sentence"] = splited_words
                l.append(d)

        SplitWords.close()
        return l

    @staticmethod
    def __del_low_df_word(splited_words_list):
        """
        删除文档频率 <= 3 的词汇，因这些词汇有可能是生僻词或表意能力比较差的词
        :return:
        """
        def fun(splited_words):
            splited_words_0 = splited_words.get("sentence")
            aa = filter(lambda t: Feature.n_contains(t, splited_words_list_0) > 3, splited_words_0)
            splited_words["sentence"] = aa

        splited_words_list_0 = [d.get("sentence") for d in splited_words_list]
        map(fun, splited_words_list)
        return [d for d in splited_words_list if d.get("sentence")]

    @staticmethod
    def union(dict1, dict2):
        """
        合并两个字典
        if has the same key, then add value
        else append {key: value}
        :param dict1: {key: [weight, frequency, ...]} or {key: weight}
        :param dict2: {key: [weight, frequency, ...]} or {key: weight}
        :return: d
        """
        d = dict(dict1)
        for k, v in dict2.items():
            if k in dict1:
                if hasattr(v, "__getitem__"):
                    d[k][0] = (d[k][0] + v[0]) / 2
                    d[k][1] += v[1]
                else:
                    d[k] = (d[k] + v) / 2
            else:
                d[k] = v
        return d

    @staticmethod
    def __print_top_key_word(res):
        all_class = Feature.all_class_text(res)
        for c in EMOTION_CLASS.keys():
            each_class = reduce(Feature.union, all_class.get(c))
            sort = sorted(each_class.items(), key=lambda x: x[1], reverse=True)
            sort = sort[0:50]
            print c
            for k, v in sort:
                print k.decode("utf-8") + ":" + str(v)
            print
        print

if __name__ == "__main__":
    ss = "我在高楼高楼上高楼"
    s = "高楼"
    print ss.count(s)
    print len(ss)
    print Feature.tf(s, ss)
    if s in ss:
        print "Yes"
