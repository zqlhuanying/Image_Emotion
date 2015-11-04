# encoding: utf-8

import cv2

from com.image.rectangle import Rectangle

__author__ = 'zql'
__date__ = '2015/10/26'


def get_rects(height, width, rows, columns):
    """
    返回分块后的每个块的信息
    return Rectangle
    """
    per_height = height / rows
    per_width = width / columns

    # 若图像无法整除，即分块后会有剩余部分，则将剩余部分的信息归入到最后一块
    # 剩余部分
    height_left = height - per_height * rows
    width_left = width - per_width * columns

    rects = list()

    for i in xrange(rows):
        is_last_row = False
        if i == rows - 1:
                last_row_height = per_height + height_left
                is_last_row = True
        for j in xrange(columns):
            is_last_column = False
            if j == columns - 1:
                last_column_width = per_width + width_left
                is_last_column = True
            x = i * per_height
            y = j * per_width
            rects.append(Rectangle(x, y,
                                   per_width if not is_last_column else last_column_width,
                                   per_height if not is_last_row else last_row_height))
    return rects


def split_img(img, rows=2, columns=2):
    """"
    将图片分成rows * columns块
    """
    if rows <= 0 or columns <= 0 or type(rows) != int or type(columns) != int:
        raise ValueError("value is invalid")

    src_height = img.shape[0]
    src_width = img.shape[1]

    # 获取分块后的矩形信息
    rects = get_rects(src_height, src_width, rows, columns)

    for index, rect in enumerate(rects):
        """ cv2.imshow("Image" + str(index), img[rect.get_start_point()[0]:rect.get_start_point()[0] + rect.get_height(),
                                             rect.get_start_point()[1]:rect.get_start_point()[1] + rect.get_width(),
                                             :])"""
        img_name = "F:\\Image" + str(index) + ".jpg"
        cv2.imwrite(img_name, img[rect.get_start_point()[0]:rect.get_start_point()[0] + rect.get_height(),
                                  rect.get_start_point()[1]:rect.get_start_point()[1] + rect.get_width(),
                                  :])
    # cv2.waitKey(0)

if __name__ == "__main__":
    src_img = cv2.imread('F:\\test.jpg')
    # src_img = cv2.imread("/home/zhuang/PythonProjects/Image_Emotion/test (2).jpg")
    split_img(src_img)
