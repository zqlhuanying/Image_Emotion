# encoding: utf-8
from compiler.ast import flatten
import random
from com import RESOURCE_BASE_URL, EMOTION_CLASS

__author__ = 'zql'
__date__ = '2015/12/24'

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def append_data(src, dest, classes):
    """
    read class c from src, append result to dest
    :param src: url
    :param dest: url
    :param classes: classes a list
    :return:
    """
    for c in classes:
        if c not in EMOTION_CLASS.keys():
            raise ValueError("%s is not support class" % c)

    src_tree = None
    dest_tree = None
    try:
        src_tree = ET.parse(src)
        dest_tree = ET.parse(dest)
    except IOError:
        print "cannot parse file"
        exit(-1)

    if src_tree and dest_tree:
        src_root = src_tree.getroot()
        dest_root = dest_tree.getroot()

        l = [src_root.findall("weibo[@emotion-type='%s']" % c) for c in classes]
        l = flatten(l)
        random.shuffle(l)

        [dest_root.append(l1) for l1 in l]

        # write to file
        dest_tree.write(dest, encoding="utf-8")

        print "append data is done."


if __name__ == "__main__":
    src_url = RESOURCE_BASE_URL + "weibo_trains.xml"
    dest_url = RESOURCE_BASE_URL + "weibo_samples.xml"
    classes = ["fear", "surprise", "anger"]
    append_data(src_url, dest_url, classes)
