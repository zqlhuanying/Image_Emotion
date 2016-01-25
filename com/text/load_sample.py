# encoding: utf-8
from __future__ import division
from compiler.ast import flatten
import random
from com import RESOURCE_BASE_URL, EMOTION_CLASS
from com.text import collect
from com.text.split_words_nlpir import SplitWords

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

__author__ = 'zql'
__date__ = '2015/11/16'


class Load:
    """
    加载微博数据，加载的数据是 unicode 编码的（但是在XML中已经指示是 UTF-8 ）
    但是分词后的数据是 UTF-8 编码的，所以为了保持一致，均采用 UTF-8 编码
    """
    def __init__(self):
        pass

    @staticmethod
    def load_training_balance():
        url = RESOURCE_BASE_URL + "weibo_samples_part.xml"
        ratio = 2 / 3
        return Load.__load(url, ratio, balance=True)

    @staticmethod
    def load_test_balance():
        url = RESOURCE_BASE_URL + "weibo_samples_part.xml"
        ratio = 1 / 3
        return Load.__load(url, ratio, direction=False, balance=True)

    @staticmethod
    def load_training_objective_balance():
        url = RESOURCE_BASE_URL + "weibo_samples.xml"
        ratio = 2 / 3
        datas = Load.__load(url, ratio, subjective=False, balance=True)
        [Load.chd_attr(data, "emotion-1-type", "%s" % "N" if data.get("emotion-1-type") == "none" else "Y")
         for data in datas]
        return datas

    @staticmethod
    def load_test_objective_balance():
        url = RESOURCE_BASE_URL + "weibo_samples.xml"
        ratio = 1 / 3
        datas = Load.__load(url, ratio, direction=False, subjective=False, balance=True)
        [Load.chd_attr(data, "emotion-1-type", "%s" % "N" if data.get("emotion-1-type") == "none" else "Y")
         for data in datas]
        return datas

    @staticmethod
    def load_incr_datas():
        url = RESOURCE_BASE_URL + "weibo_samples_incr.xml"
        ratio = 1 / 2
        incr_train_datas = Load.__load(url, ratio, balance=True)
        return incr_train_datas
#        SplitWords.__init__()
#        incr_train_datas = [SplitWords.split_words(data.get("sentence")) for data in collect.read_weibo("collect/incr")]
#        SplitWords.close()
#        incr_train_datas = [{d: data.count(d) for d in set(data)} for data in incr_train_datas]
#        return incr_train_datas

    @staticmethod
    def __load(url, ratio, direction=True, subjective=True, balance=False):
        """
        Loading Training Data Except Objective Sentence
        :param url:
        :param direction: 默认从上往下取
        :param subjective: 加载主观句还是客观句，默认加载主观数据
                            True: 加载多类别，即情绪标签
                            False: 加载二类别，即主客观
        :param balance: 是否需要平衡加载数据集，默认以非平衡的方式加载
        :return:
        """
        # 若是加载客观的数据，也就没有平衡加载的概念
#        if not subjective and balance:
#            raise AttributeError("can not load data which is objective and use balanced way!")

        tree = None
        try:
            tree = ET.parse(url)
        except IOError:
            print "cannot parse file"
            exit(-1)
        if tree is not None:
            # get the root
            root = tree.getroot()

            # get the direct child
            # 若非平衡加载，只需要将所有的 weibo 看成一类，即可复用代码
            # todo
            # ElementTree XPath 貌似不支持 not、!= 操作，所有暂时采用以下方案代替
            each_class = [[sentence for sentence in root.findall("weibo") if sentence.get("emotion-type") != "none"]]
            if not subjective:
                each_class = [root.findall("weibo[@emotion-type]")]

            if balance:
                each_class = [root.findall("weibo[@emotion-type='%s']" % c) for c in EMOTION_CLASS.keys()]
                if not subjective:
                    each_class = Load.partition(root.findall("weibo[@emotion-type]"),
                                                lambda x: x.get("emotion-type") == "none")

            each_class_size = [len(c) for c in each_class]
            each_class_range = [slice(int(n * ratio)) for n in each_class_size]
            if not direction:
                _reverse_ratio = 1 - ratio
                each_class_range = [slice(int(n * _reverse_ratio), n) for n in each_class_size]

            sentences = []
            for i, each in enumerate(each_class):
                _range = each_class_range[i]
                sentences.append([Load.integrate(sentence) for sentence in each[_range]])

            # shuffle
            sentences = flatten(sentences)
            # random.shuffle(sentences)

            return [{"sentence": sentence.text.encode("utf_8"),
                     "emotion-tag": sentence.get("emotion_tag"),
                     "emotion-1-type": sentence.get("emotion-type"),
                     "emotion-2-type": sentence.get("emotion-2-type")}
                    for sentence in sentences]

#    @staticmethod
#    def __load(url, ratio, direction=True, subjective=True, balance=False):
#        """
#        Loading Training Data Except Objective Sentence
#        :param url:
#        :param direction: 默认从上往下取
#        :param subjective: 加载主观句还是客观句
#        :param balance: 是否需要平衡加载数据集
#        :return:
#        """
#        tree = None
#        try:
#            tree = ET.parse(url)
#        except IOError:
#            print "cannot parse file"
#            exit(-1)
#        if tree is not None:
#            # get the root
#            root = tree.getroot()
#
#            # get the direct child
#            direct_childs = root.findall("weibo")
#
#            # range
#            _range = slice(int(len(direct_childs) * ratio))
#            if not direction:
#                _range = slice(int(len(direct_childs) * (1 - ratio)), len(direct_childs), None)
#
#            sentences = [Load.integrate(child) for child in direct_childs[_range]]
#
#            # 返回训练集中属于主观句的部分
#            return [{"sentence": sentence.text.encode("utf_8"),
#                     "emotion-tag": sentence.get("emotion_tag"),
#                     "emotion-1-type": sentence.get("emotion-type"),
#                     "emotion-2-type": sentence.get("emotion-2-type")}
#                    for sentence in flatten(sentences)
#                    if sentence.get("emotion-type") != "none"]
#
#            sentences = [child.findall("sentence") for child in direct_childs[_range]]
#            # 将 emotion-2-type 的部分也 add 进来
#            sentences += [Load.chd_a_little(child) for child in flatten(sentences)
#                          if child.get("emotion-2-type") and child.get("emotion-2-type") != "none"]
#
#            # 返回训练集中属于主观句的部分
#            return [{"sentence": sentence.text.encode("utf_8"),
#                     "emotion-tag": sentence.get("emotion_tag"),
#                     "emotion-1-type": sentence.get("emotion-1-type"),
#                     "emotion-2-type": sentence.get("emotion-2-type")}
#                    for sentence in flatten(sentences)
#                    if sentence.get("emotion_tag") == "Y"]

    @staticmethod
    def integrate(sentence):
        sentence.text = "".join([text for text in sentence.itertext() if text.strip()])
        return sentence

    @staticmethod
    def chd_attr(element, attr0, value):
        if attr0 not in element:
            raise ValueError("wrong attibute")
        element[attr0] = value

    @staticmethod
    def partition(iterable, condition):
        """
        separate a Python list into two lists, according to condition
        :param iterable:
        :param condition:
        :return:
        """
        def partition_element(partitions, element):
            (partitions[0] if condition(element) else partitions[1]).append(element)
            return partitions
        return reduce(partition_element, iterable, [[], []])

if __name__ == "__main__":
    s = Load.load_training_balance()
    s1 = Load.load_training_objective_balance()
    print len(s)
