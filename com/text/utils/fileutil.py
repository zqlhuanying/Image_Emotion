# encoding: utf-8
import os

__author__ = 'zql'
__date__ = '2015/12/15'


class FileUtil:
    def __init__(self):
        pass

    @staticmethod
    def isexist(path):
        """
        判断文件或目录是否存在
        :param path:
        :return:
        """
        return os.path.exists(path)

    @staticmethod
    def isempty(path):
        """
        判断文件或目录是否为空
        文件或目录必须存在
        :param path:
        :return:
        """
        if not FileUtil.isexist(path):
            raise ValueError("file or dir must be exist")
        if FileUtil.isfile(path):
            return len(open(path).read()) == 0
        else:
            return len(FileUtil.listdir(path)) == 0

    @staticmethod
    def isfile(path):
        """
        判断是否是文件
        isfile: 前提是必须存在
        :param path: 必须存在的路径，否则均返回false
        :return:
        """
        return os.path.isfile(path)

    @staticmethod
    def getparentdir(path):
        """
        取得父目录
        :param path:
        :return:
        """
        return os.path.dirname(path)

    @staticmethod
    def mkdirs(path):
        """
        创建目录
        :param path:
        :return:
        """
        if not FileUtil.isexist(path):
            os.makedirs(path)

    @staticmethod
    def listdir(path, isrecursion=True):
        """
        列出目录下所有的文件
        :param path:
        :param isrecursion: 是否递归子目录
        :return:
        """
        if FileUtil.isfile(path):
            raise ValueError("must be a dir")
        l = []
        for parent, dirnames, filenames in os.walk(path):
            for filename in filenames:
                l.append(os.path.join(parent, filename))
            if not isrecursion:
                break
        return l

    @staticmethod
    def write(path, data):
        FileUtil.mkdirs(path)

        s = len(data)
        fp = open(path, "w")
        for i, d in enumerate(data):
            sentence = d["sentence"]
            if sentence:
                if isinstance(sentence, list):
                    fp.write(",".join(sentence))
                    fp.write(",")
                else:
                    [fp.write(t[0] + ":" + str(t[1]) + ",") for t in sentence.items()]
                fp.write(d["emotion-1-type"])
                if i < s - 1:
                    fp.write("\n")
        fp.close()

    @staticmethod
    def read(path):
        def try_list_2_dict(l):
            if l[0].find(":") > 0:
                d = {}
                map(lambda x: d.setdefault(x.split(":")[0], float(x.split(":")[1])), l)
                return d
            return l

        def process(line):
            d = {}
            l = line.split(",")
            d["emotion-1-type"] = l.pop()
            d["sentence"] = try_list_2_dict(l)
            return d
        return [process(line.strip("\n")) for line in open(path)]


if __name__ == "__main__":
    p = "/home/zhuang/1/11/111"
    FileUtil.mkdirs(p)
    p = "/home/zhuang/PythonWorkspace/Image_Emotion/com/resource"
    FileUtil.listdir(p, True)
    p = "/home/zhuang/1/11/111.txt"
    print FileUtil.isfile(p)
