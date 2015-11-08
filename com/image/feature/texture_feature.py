#  encoding: utf-8
from com.image.feature.feature import Feature

__author__ = 'root'
__date__ = '15-11-8'


class TextureFeature(Feature):
    """
    图像纹理特征
    """
    def __init__(self, image):
        super(TextureFeature, self).__init__(image)

    def cal_feature(self):
        pass
