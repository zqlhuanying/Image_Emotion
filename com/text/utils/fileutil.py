# encoding: utf-8
import os

__author__ = 'zql'
__date__ = '2015/12/15'


class FileUtil:
    def __init__(self):
        pass

    @staticmethod
    def isexist(path):
        return os.path.exists(path)

    @staticmethod
    def isempty(path):
        return len(open(path).read()) == 0

    @staticmethod
    def write(path, data):
        parent_dir = os.path.dirname(path)
        if not FileUtil.isexist(parent_dir):
            os.makedirs(parent_dir)

        s = len(data)
        fp = open(path, "w")
        for i, d in enumerate(data):
            sentence = d["sentence"]
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


