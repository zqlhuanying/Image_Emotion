# encoding: utf-8
import base64
import os
import requests

import time
import urllib
from compiler.ast import flatten

from com import RESOURCE_BASE_URL
from com.text.utils.fileutil import FileUtil

__author__ = 'root'
__date__ = '16-01-15'


def collect_weibo():
    # thumbnail_pic: 所发的微博中第一张配图的 url
    # 要想获取当前微博下所有的配图，使用 pic_urls
    # 微博下不同尺寸的图片，文件名是一致的，只是放在了不同的目录
    # thumbnail: 小图
    # bmiddle: 中图
    # large: 大图
    def handle_write(data):
        for d in data.get("statuses"):
            yield ("sentence:" + d["text"] + "\n" +
                   "img:" + process_img(d.get("pic_urls")) + "\n").encode("utf-8")

    def process_img(urls):
        return ",".join([__process_img(u.get("thumbnail_pic", "")) for u in urls])

    print "Collecting WeiBo: ", time.strftime('%Y-%m-%d %H:%M:%S')

    appkey = "2300736086"
    url = "https://api.weibo.com/2/statuses/public_timeline.json"
    count = 200
    theurl = url + "?source=%s" % appkey + "&count=%s" % count

    json = _collect(theurl)

    if "error" in json:
        raise ValueError("Collecting error")

    path = RESOURCE_BASE_URL + "collect"
    current = time.strftime('%Y-%m-%d %H:%M:%S')
    path = path + os.sep + current + ".txt"

    write_to_file(path, json, handle_write)
    print "Collecting Done: ", time.strftime('%Y-%m-%d %H:%M:%S')


def collect_emotion():
    def handle_write(data):
        for d in data:
            yield ("category:" + d["category"] + "," +
                   "img:" + __process_img(d.get("icon", "")) + "," +
                   "phrase:" + d.get("phrase") + "\n").encode("utf-8")

    print "Collecting Emotion: ", time.strftime('%Y-%m-%d %H:%M:%S')

    appkey = "2300736086"
    url = "https://api.weibo.com/2/emotions.json"
    theurl = url + "?source=%s" % appkey

    json = _collect(theurl)

    if "error" in json:
        raise ValueError("Collecting error")

    dir_ = RESOURCE_BASE_URL + "collect/emotion"
    current = time.strftime('%Y-%m-%d %H:%M:%S')
    path = dir_ + "/" + current + ".txt"

    write_to_file(path, json, handle_write)
    print "Collecting Done: ", time.strftime('%Y-%m-%d %H:%M:%S')


def read_weibo(path, isreadimg=False):
    def handle_read(datas):
        fit_datas = datas
        if not isreadimg:
            fit_datas = [data for data in datas if not data.startswith("img")]

        l = []
        d = dict()
        for data in fit_datas:
            if data.startswith("sentence"):
                d = dict()
                d["sentence"] = data[data.find(":") + 1:]
                l.append(d)
            elif data.startswith("img"):
                d["img"] = filter(lambda x: x, data[data.find(":") + 1:].split(","))
        return l

    path = path if path.startswith(RESOURCE_BASE_URL) else os.path.join(RESOURCE_BASE_URL, path)
    filenames = FileUtil.listdir(path, isrecursion=False)
    return flatten([read_from_file(filename, handle_read) for filename in filenames])


def _collect(url, auth="Basic"):
    """
    在微博中有两种授权方式：1.OAuth2.0 2.Basic Auth
    Basic Auth 参考地址: http://www.cnblogs.com/QLeelulu/archive/2009/11/22/1607898.html
    同时微博中有些 API 可以匿名访问 参考地址: http://www.cnitblog.com/ldqok/archive/2011/04/01/73232.html
    :return:
    """
    header = {}
    if auth == "Basic":
        authheader = basicauth()
        header = {"Authorization": authheader}

    r = requests.get(url, headers=header)
    return r.json()


def __process_img(url):
    filepath = ""
    index = url.rfind("/")
    if index > -1:
        filename = url[index + 1:]
        dir_ = RESOURCE_BASE_URL + "img"
        FileUtil.mkdirs(dir_)
        filepath = dir_ + os.sep + filename
        urllib.urlretrieve(url, filepath)

    return filepath


def write_to_file(path, data, call):
    parent_dir = FileUtil.getparentdir(path)
    FileUtil.mkdirs(parent_dir)

    with open(path, "w") as fp:
        s = call(data)
        if isinstance(s, basestring):
            fp.write(s)
        else:
            [fp.write(s0) for s0 in s]


def read_from_file(path, call):
    with open(path) as fp:
        l = [line.strip("\n") for line in fp.readlines()]
        return call(l)


def basicauth():
    username = "1416091730@qq.com"
    password = "zql330327ZQL"
    # 注意哦，这里最后会自动添加一个\n
    base64string = base64.encodestring('%s:%s' % (username, password))[: -1]
    authheader = "Basic %s" % base64string

    return authheader

    # 调用方法
#    theurl = "https://api.weibo.com/2/statuses/public_timeline.json"
#    r = requests.get(theurl, headers={"Authorization": authheader})
#    print r.json()

if __name__ == "__main__":
    collect_weibo()
    # collect_emotion()
    # l = read_weibo("collect/incr")
    print
