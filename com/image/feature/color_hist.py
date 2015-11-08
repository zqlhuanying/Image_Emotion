# encoding: utf-8
import cv2
import numpy as np

from color_feature import ColorFeature
from com.image.constant.constant import Constant

__author__ = 'zql'
__date__ = '2015/11/3'


class ColorHist(ColorFeature):
    """
    颜色直方图
    """
    def __init__(self, image):
        super(ColorHist, self).__init__(image)

    def cal_feature(self):
        """颜色直方图特征"""
        src_img = super(ColorHist, self).bgr2hsv()

        h, s, v = cv2.split(src_img)

        hist_img_h = self.color_hist(h)
        hist_img_s = self.color_hist(s)
        hist_img_v = self.color_hist(v)
        cv2.imshow("H", hist_img_h)
        cv2.imshow("S", hist_img_s)
        cv2.imshow("V", hist_img_v)
        cv2.waitKey(0)

    def color_hist(self, src_img):
        """计算颜色直方图"""
        # 画图的颜色
        color = [255, 0, 0]
        # 背景图
        hist_img = np.zeros((256, 256, 3), dtype=np.uint8)
        # 直方图
        hist = cv2.calcHist([src_img], [0], None, [256], [0, 256])
        # 归一化
        cv2.normalize(hist, hist, 0, 255 * 0.9, cv2.NORM_MINMAX)
        # 归一化后为浮点数，取整操作
        hist_int = np.int32(np.around(hist))
        for h in range(256):
            cv2.line(hist_img, (h, 256), (h, 256 - hist_int[h]), color)

        return hist_img


if __name__ == "__main__":
    imgname = Constant.BASE_URL + "test3.jpg"
    ColorHist(imgname).cal_feature()
