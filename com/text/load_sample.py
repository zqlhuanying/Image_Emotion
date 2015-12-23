# encoding: utf-8
from __future__ import division
from compiler.ast import flatten
from com import RESOURCE_BASE_URL

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
    def load_training():
        url = RESOURCE_BASE_URL + "weibo_samples.xml"
        ratio = 2 / 3
        return Load.__load(url, ratio)

    @staticmethod
    def load_test():
        url = RESOURCE_BASE_URL + "weibo_samples.xml"
        ratio = 1 / 3
        return Load.__load(url, ratio, False)

    @staticmethod
    def __load(url, ratio, direction=True):
        """
        Loading Training Data Except Objective Sentence
        :param url:
        :param direction: 默认从上往下取
        :return:
        """
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
            direct_childs = root.findall("weibo")

            # range
            _range = slice(int(len(direct_childs) * ratio))
            if not direction:
                _range = slice(int(len(direct_childs) * (1 - ratio)), len(direct_childs), None)

            sentences = [Load.chd_a_little_1(child) for child in direct_childs[_range]]

            # 返回训练集中属于主观句的部分
            return [{"sentence": sentence.text.encode("utf_8"),
                     "emotion-tag": sentence.get("emotion_tag"),
                     "emotion-1-type": sentence.get("emotion-type"),
                     "emotion-2-type": sentence.get("emotion-2-type")}
                    for sentence in flatten(sentences)
                    if sentence.get("emotion-type") != "none"]
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
    def chd_a_little_1(sentence):
        sentence.text = "".join([text for text in sentence.itertext() if text.strip()])
        return sentence
        # print sentence.text
        # print

    @staticmethod
    def chd_a_little(sentence):
        attribute = sentence.attrib
        e = ET.Element("sentence", attribute)
        e.text = sentence.text
        e.attrib["emotion-1-type"] = e.attrib["emotion-2-type"]
        return e


if __name__ == "__main__":
    s = Load.load_training()
    print len(s)
