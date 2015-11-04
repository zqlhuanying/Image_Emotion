# encoding: utf-8
import cv2
import numpy as np

from com.image.constant.constant import Constant
from color_feature import ColorFeature

__author__ = 'zql'
__date__ = '2015/11/3'


class ColorHist(ColorFeature):
    """颜色直方图"""
    def __init__(self):
        pass

    def cal_feature(self, image):
        """颜色直方图特征"""
        # 输出的顺序为BGR
        src_img = cv2.imread(image)
        # img_array_to_file(Constant.BASE_URL + "RGB.txt", src_img)

        # 将RGB转换成HSV
        # H : COLOR_BGR2HSV [0, 179]; COLOR_BGR2HSV_FULL [0, 255]
        # S : [0, 255]
        # V : [0, 255]
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV_FULL)
        # img_array_to_file(Constant.BASE_URL + "HSV.txt", src_img)

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

    def print_img_array(self, array):
        """打印图像数组"""
        shape = array.shape
        last_dim_size = shape[len(shape) - 1]

        for index, value in np.ndenumerate(array):
            # print 函数默认是换行的，想不换行输出，在 print 后加 ，
            # 也可以使用sys.stdout.write
            print value,

            if index[len(index) - 1] == last_dim_size - 1:
                print("\n"),
            else:
                print "",

    def img_array_to_file(self, filename, array):
        """将图像数组输出到文件"""
        shape = array.shape
        last_dim_size = shape[len(shape) - 1]

        fp = open(filename, "w")

        for index, value in np.ndenumerate(array):
            fp.write(str(value))
            if index[len(index) - 1] == last_dim_size - 1:
                fp.write("\n")
            else:
                fp.write(" ")

        fp.close()

if __name__ == "__main__":
    imgname = "f:\\3.jpg"
    ColorHist().cal_feature(imgname)
