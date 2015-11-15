# encoding: utf-8
import platform
import regex
import string
import pynlpir.nlpir as nlpir
from com.text.constant import RESOURCE_BASE_URL

__author__ = 'zql'
__date__ = '2015/11/10'


class SplitWords:
    """
    中文分词
    """
    @staticmethod
    def __init__():
        # 初始化nlpir资源
        if not nlpir.Init(nlpir.PACKAGE_DIR, nlpir.UTF8_CODE, None):
            print "Initialize NLPIR failed"
            exit(-1)

    @staticmethod
    def split_words(s):
        # 去除标签

        # 去除标点符号
        s = SplitWords.__del_punctuation(s)

        # 去除数字
        s = SplitWords.__del_digit(s)

        # 分词
        words = nlpir.ParagraphProcess(s, False)

        # 去掉左右两边多余的空格，并分割
        words = words.strip().split(" ")

        # 去掉中文停用词
        # 不管分词后的结果是否带有词性
        words = SplitWords.__del_stop(words, SplitWords.__read_chinese_stoplist())
        # 此方法只能是分词后不带词性才可以使用
        # words = [word for word in words if word not in SplitWords.__read_chinese_stoplist()]

        # 去掉英文停用词
        words = SplitWords.__del_stop(words, SplitWords.__read_english_stoplist())

        encoding = SplitWords.__get_encoding()

#        for index, word in enumerate(words):
#            if isinstance(word, unicode):
#                words[index] = word.encode(encoding)
#            else:
#                words[index] = word.decode("utf_8").encode(encoding)

        return words

    @staticmethod
    def close():
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

    @staticmethod
    def __del_punctuation(s):
        # todo
        # 不要删除表情和表情符号
        """
        去除标点符号，不管是英文标点还是中文标点
        :return:
        """
        """
        去除英文标点和空格
        翻译表
        string.translate 将字符串先过滤，后使用翻译表，将相应的字符翻译成对应的字符
        """
        translate_table = string.maketrans(",", ",")
        del_english_punctuation = string.punctuation + " "
        s = string.translate(s, translate_table, del_english_punctuation)

        """
        去除中文标点
        Python 自带的 re 模块貌似不支持 unicode 属性，因此需要安装 regex 模块
        这样就可以使用 \p{} 这样的 unicode 属性
        使用时，正则表达式和字符串都必须经过 unicode 编码
        """
        s = regex.sub(ur"\p{P}|\p{S}", "", s if isinstance(s, unicode) else s.decode("utf_8"))
        return s.encode("utf_8")

    @staticmethod
    def __del_digit(s):
        return regex.sub(r"\d", "", s)

    @staticmethod
    def __del_stop(words, stoplist):
        result = list()
        for word in words:
            pos_tag = None
            if word.find("/") > 0:
                pos_tag = word[word.find("/"):]
                word = word[:word.find("/")]
            if word not in stoplist:
                if pos_tag is not None:
                    word = word + pos_tag
                result.append(word)
        return result

    @staticmethod
    def __read_chinese_stoplist():
        url = RESOURCE_BASE_URL + "chinese_stoplist.txt"
        # 下面这行使用方便，但觉得会不会不会关闭打开文件后的资源
        # return [line.strip("\n") for line in open(url).readlines()]
        fp = open(url)
        stoplist = [line.strip("\n") for line in fp.readlines()]
        fp.close()
        return stoplist

    @staticmethod
    def __read_english_stoplist():
        url = RESOURCE_BASE_URL + "english_stoplist.txt"
        fp = open(url)
        stoplist = [line.strip("\n") for line in fp.readlines()]
        fp.close()
        return stoplist

if __name__ == "__main__":
    string1 = 'NLPIR分词系a统前身《为？2000年&@发布的<><><><>ICTCLAS词法分1885析系统,从2009年开始,' \
              '为了和以前工作进行大的区隔，并推广NLPIR自然语言处理与信息检索共享'
    SplitWords.__init__()
    splited_words = SplitWords.split_words(string1)
    SplitWords.close()

    print len(splited_words)
    for splited_word in splited_words:
        print splited_word == "系统"
        if isinstance(splited_word, unicode):
            print splited_word.encode("utf_8")
        else:
            print splited_word.decode("utf_8").encode("utf_8")
