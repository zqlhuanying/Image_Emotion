# encoding: utf-8

__author__ = 'zql'
__date__ = '2015/10/26'


class Rectangle:
    """
    矩形类，描诉：左上角点的坐标x,y 宽度width，高度height
    """
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_start_point(self):
        return self.x, self.y

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

if __name__ == "__main__":
    print(Rectangle.__doc__.decode("UTF-8"))
