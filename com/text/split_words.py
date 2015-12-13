# encoding: utf-8
import regex
import pynlpir.nlpir as nlpir
from com import RESOURCE_BASE_URL

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
        s = SplitWords.__del_non_tag(s)

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

        # 去掉多余的空格
        words = SplitWords.__del_blank(words)

        return words

    @staticmethod
    def close():
        nlpir.Exit()

    @staticmethod
    def __del_non_tag(s):
        # 去除微博中无用的标签
        def theme_tag():
            # 主题标签
            return regex.sub(r"#.*?#", "", s)

        def tran_tag():
            # //@__: 转发标签
            return regex.sub(r"//@.*?:", "", s)

        def url_tag():
            # url 标签
            return regex.sub(r"(http|https)?://([a-zA-Z|\d]+\.)+[a-zA-Z|\d]+(/[a-zA-Z|\d|\-|\+|_./?%=]*)?", "", s)

        def alt_tag():
            # @提醒标签
            return regex.sub(ur"@[\u4e00-\u9fa5\w-]{1,30}(?=\s|:)", "",
                             s if isinstance(s, unicode) else s.decode("utf_8"))

        s = theme_tag()
        s = tran_tag()
        s = url_tag()
        s = alt_tag()
        return s.encode("utf_8")

    @staticmethod
    def __del_punctuation(s):
        # todo
        # 不要删除表情和表情符号
        # 表情符号暂时不考虑 *^_^*
        """
        去除标点符号，不管是英文标点还是中文标点
        :return:
        """
        """
        Unicode 编码并不只是为某个字符简单定义了一个编码，而且还将其进行了归类。
        \pP 其中的小写 p 是 property 的意思，表示 Unicode 属性，用于 Unicode 正表达式的前缀。
        大写 P 表示 Unicode 字符集七个字符属性之一：标点字符。
        其他六个是
        L：字母；
        M：标记符号（一般不会单独出现）；
        Z：分隔符（比如空格、换行等）；
        S：符号（比如数学符号、货币符号等）；
        N：数字（比如阿拉伯数字、罗马数字等）；
        C：其他字符

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
            if word.lower() not in stoplist:
                if pos_tag is not None:
                    word = word + pos_tag
                result.append(word)
        return result

    @staticmethod
    def __del_blank(words):
        return [word for word in words if word != ""]

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
    string1 = r'//@王久辛:@王久辛 @王久辛 留大量[愤怒] 预习与@王久辛:作业,让家长@hj 给孩子报铺导#辅导班#班,他们再#《》赚一把钱~~美特伺邦威周成健: 做 ME AND CITY是非常有价值的。' \
              r'群众的呼声VERY very importantly！！！[大笑] [大笑] *^_^*'
    SplitWords.__init__()
    splited_words = SplitWords.split_words(string1)
    SplitWords.close()
    print len(splited_words)
    for splited_word in splited_words:
        if isinstance(splited_word, unicode):
            print splited_word
        else:
            print splited_word.decode("utf_8")