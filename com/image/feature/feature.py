# encoding: utf-8
import cv2

__author__ = 'zql'
__date__ = '2015/11/4'


class Feature(object):
    """
    所有特征的父类
    """
    def __init__(self, image):
        self.image = image

    def cal_feature(self):
        pass

    def bgr2hsv(self):
        # 输出的顺序为BGR
        src_img = cv2.imread(self.image)
        # CommonUtil.img_array_to_file(Constant.BASE_URL + "RGB.txt", src_img)

        # 将RGB转换成HSV
        # H : COLOR_BGR2HSV [0, 179]; COLOR_BGR2HSV_FULL [0, 255]
        # S : [0, 255]
        # V : [0, 255]
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV_FULL)
        # CommonUtil.img_array_to_file(Constant.BASE_URL + "HSV.txt", src_img)
        return src_img
