# encoding: utf-8
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
    加载微博数据
    """
    def __init__(self):
        pass

    @staticmethod
    def load_training(url):
        """
        Loading Training Data Except Objective Sentence
        :param url:
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

            # 训练集数目 占总数的 2/3
            max_training = len(direct_childs) * 2 / 3

            sentences = [child.findall("sentence") for child in direct_childs[:max_training]]

            # 返回训练集中属于主观句的部分
            return [{"sentence": sentence.text,
                     "emotion-tag": sentence.get("emotion_tag"),
                     "emotion-1-type": sentence.get("emotion-1-type"),
                     "emotion-2-type": sentence.get("emotion-2-type")}
                    for sentence in flatten(sentences)
                    if sentence.get("emotion_tag") == "Y"]

if __name__ == "__main__":
    url = RESOURCE_BASE_URL + "weibo_samples.xml"
    s = Load.load_training(url)
    print len(s)
