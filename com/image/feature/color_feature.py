# encoding: utf-8

from feature import Feature

__author__ = 'zql'
__date__ = '2015/11/4'


class ColorFeature(Feature):
    """
    图像颜色特征
    """
    def __init__(self, image):
        super(ColorFeature, self).__init__(image)

    def cal_feature(self):
        pass
