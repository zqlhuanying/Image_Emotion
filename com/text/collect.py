# encoding: utf-8
import base64
import os
import requests

import time
import urllib

from com import RESOURCE_BASE_URL
from com.text.utils.fileutil import FileUtil

__author__ = 'root'
__date__ = '16-01-15'


def collect_weibo():
    def write_to_file(data):
        # thumbnail_pic: 所发的微博中第一张配图的 url
        # 要想获取当前微博下所有的配图，使用 pic_urls
        # 微博下不同尺寸的图片，文件名是一致的，只是放在了不同的目录
        # thumbnail: 小图
        # bmiddle: 中图
        # large: 大图
        dir_ = RESOURCE_BASE_URL + "collect"
        current = time.strftime('%Y-%m-%d %H:%M:%S')
        path = dir_ + "/" + current + ".txt"

        if not FileUtil.isexist(dir_):
            os.makedirs(dir_)

        with open(path, "w") as fp:
            [fp.write(("sentence:" + d["text"] + "," +
                       "img:" + __process_img(d.get("thumbnail_pic", "")) + "\n").encode("utf-8")) for d in data["statuses"]]

    print "Collecting WeiBo: ", time.strftime('%Y-%m-%d %H:%M:%S')

    appkey = "2300736086"
    url = "https://api.weibo.com/2/statuses/public_timeline.json"
    count = 200
    theurl = url + "?source=%s" % appkey + "&count=%s" % count

    json = _collect(theurl)

    if "error" in json:
        raise ValueError("Collecting error")

    write_to_file(json)
    print "Collecting Done: ", time.strftime('%Y-%m-%d %H:%M:%S')


def collect_emotion():
    def write_to_file(data):
        dir_ = RESOURCE_BASE_URL + "collect/emotion"
        current = time.strftime('%Y-%m-%d %H:%M:%S')
        path = dir_ + "/" + current + ".txt"

        if not FileUtil.isexist(dir_):
            os.makedirs(dir_)

        with open(path, "w") as fp:
            [fp.write(("category:" + d["category"] + "," +
                       "img:" + __process_img(d.get("icon", "")) + "," +
                       "phrase:" + d.get("phrase") + "\n").encode("utf-8")) for d in data]

    print "Collecting Emotion: ", time.strftime('%Y-%m-%d %H:%M:%S')

    appkey = "2300736086"
    url = "https://api.weibo.com/2/emotions.json"
    theurl = url + "?source=%s" % appkey

    json = _collect(theurl)

    if "error" in json:
        raise ValueError("Collecting error")

    write_to_file(json)
    print "Collecting Done: ", time.strftime('%Y-%m-%d %H:%M:%S')


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
        filepath = dir_ + "/" + filename
        if not FileUtil.isexist(dir_):
            os.makedirs(dir_)
        urllib.urlretrieve(url, filepath)

    return filepath


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
