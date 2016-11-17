# encoding: utf-8
import math

import cv2
import os
import numpy as np

from com.constant.constant import TEST_RESOURCE
from com.image.feature.color_feature import ColorFeature

__author__ = 'zql'
__date__ = '2015/11/6'


class ColorMoment(ColorFeature):
    """
    颜色矩特征
    """
    def __init__(self, image):
        super(ColorMoment, self).__init__(image)

    def cal_feature(self):
        src_img = super(ColorMoment, self).bgr2hsv()

        h, s, v = cv2.split(src_img)

        res = []
        for obj in (h, s, v):
            res.append(self.cal_mean(obj))
            res.append(self.cal_variance(obj))
            res.append(self.cal_skewness(obj))

        return res

    def cal_mean(self, array):
        """
        计算一阶矩
        :param array: 只允许是二维数组
        :return:
        """
        if len(array.shape) != 2:
            raise AttributeError("array is not two dimensions")
        s = self.__cal_sum(array)
        return s / (array.shape[0] * array.shape[1] * 1.0)

    def cal_variance(self, array):
        """
        计算二阶矩
        :param array: 只允许是二维数组
        :return:
        """
        mean = self.cal_mean(array)
        s = self.__cal_sum(array, mean, 2)
        return math.pow(s / (array.shape[0] * array.shape[1] * 1.0), 1.0/2)

    def cal_skewness(self, array):
        """
        计算三阶矩
        :param array:
        :return:
        """
        mean = self.cal_mean(array)
        s = self.__cal_sum(array, mean, 3)
        if s < 0:
            return math.pow(math.fabs(s) / (array.shape[0] * array.shape[1] * 1.0), 1.0/3) * (-1.0)
        return math.pow(math.fabs(s) / (array.shape[0] * array.shape[1] * 1.0), 1.0/3) * 1.0

    def __cal_sum(self, array, mean=0, exp=1):
        """
        计算每个值与mean的差的exp方
        :param array:
        :param mean:
        :param exp:
        :return:
        """
        s = 0
        for index, value in np.ndenumerate(array):
            s += math.pow(value - mean, exp)
        return s

if __name__ == "__main__":
    imgname = os.path.join(TEST_RESOURCE, "photo.png")
    colorFeature = ColorMoment(imgname).cal_feature()
    print colorFeature
