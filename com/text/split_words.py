# encoding: utf-8
import platform
import pynlpir.nlpir as nlpir

__author__ = 'zql'
__date__ = '2015/11/10'


class SplitWords:
    """
    中文分词
    """

    def __init__(self):
        # 初始化nlpir资源
        if not nlpir.Init(nlpir.PACKAGE_DIR, nlpir.UTF8_CODE, None):
            print "Initialize NLPIR failed"
            exit(-1)

    def split_words(self, s):
        words = nlpir.ParagraphProcess(s, True)
        # 去掉左右两边多余的空格，并分割
        words = words.strip().split(" ")

        encoding = SplitWords.__get_encoding()

        for word in words:
            if isinstance(word, unicode):
                print word.encode(encoding)
            else:
                print word.decode("utf_8").encode(encoding)

    def close(self):
        nlpir.Exit()

    @staticmethod
    def __get_system_platform():
        return platform.system()

    @staticmethod
    def __get_encoding():
        """
        Linux 下中文需要utf_8; Windows 下中文需要gbk 或 gb2312
        :return:
        """
        system_platform = SplitWords.__get_system_platform()

        if system_platform == "Linux":
            return "utf_8"
        else:
            return "gb2312"

if __name__ == "__main__":
    string1 = 'NLPIR分词系统前身为2000年发布的ICTCLAS词法分析系统，从2009年开始，为了和以前工作进行大的区隔，并推广NLPIR自然语言处理与信息检索共享'
    split_words = SplitWords()
    split_words.split_words(string1)
    split_words.close()
