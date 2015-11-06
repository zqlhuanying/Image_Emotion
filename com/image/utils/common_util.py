# encoding: utf-8
import numpy as np

__author__ = 'zql'
__date__ = '2015/11/6'


class CommonUtil:
    """
    常用工具类
    """
    def __init__(self):
        pass

    @staticmethod
    def print_img_array(array):
        """
        打印图像数组
        """
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

    @staticmethod
    def img_array_to_file(filename, array):
        """
        将图像数组输出到文件
        """
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
