# encoding: utf-8
import os
from compiler.ast import flatten

from com.constant.constant import RESOURCE_BASE_URL
from com.text import collect
from com.text.utils.fileutil import FileUtil

__author__ = 'zql'
__date__ = '16-01-22'


def clear_un_img():
    # 图片存放的路径
    all_img_url = os.path.join(RESOURCE_BASE_URL, "img")
    # 这个目录下是需要保留的图片
    leave_img_url = os.path.join(RESOURCE_BASE_URL, "collect")

    if FileUtil.isempty(leave_img_url):
        FileUtil.empty(all_img_url)
    else:
        all_imgs = FileUtil.listdir(all_img_url)

        dirs = [leave_img_url]
        for parent, dirnames, filenames in os.walk(leave_img_url):
            for dirname in dirnames:
                dirs.append(os.path.join(parent, dirname))

        leave_imgs = []
        for dir_ in dirs:
            imglist = collect.read_weibo(dir_, isreadimg=True)
            imglist = flatten([img.get("img") for img in imglist if img.get("img")])
            leave_imgs += imglist

        # 删除多余的图片
        map(lambda p: os.remove(p) if p not in leave_imgs else None, all_imgs)


def count_img():
    # 图片存放的路径
    all_img_url = os.path.join(RESOURCE_BASE_URL, "img")
    print "It's have %d images" % len(FileUtil.listdir(all_img_url))

if __name__ == "__main__":
    clear_un_img()
    count_img()
