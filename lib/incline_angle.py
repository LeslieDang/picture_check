# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/10/12 16:53

import cv2
import numpy as np

# 1、获取图像的黑白图
def thresh_process(image_array):
    """
    对图像进行二值化处理（转为黑白图）
        读取图像
        进行灰度处理
        二值化为黑白图像
    :param image_array: 图像array数据
    :return: 返回黑白图像tresh
    """

    # 将图片转化为灰度图
    img_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # 将图像二值化为黑白图片，1表示大于阈值的变为0，否则变为最大值255
    ret, thresh = cv2.threshold(img_gray, 127, 255, 1)  # (输入的灰度图像，阈值，最大值，划分时使用的算法)

    return thresh

# 2、寻找图像中物体的轮廓
def contours_acquire(image_array):
    """
    寻找图像中物体的轮廓
    :param image_array: 图像array数据
    :return: list 返回图像的轮廓集合
    """
    thresh = thresh_process(image_array)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours: 轮廓的点集  hierarchy: 各层轮廓的索引

    contours = np.vstack((contours))
    return contours

# 4、计算图像轮廓的倾斜角度
def calculate_angle(image_array):
    """
    获取最大的三个轮廓的倾斜角度，并调整角度值
    :param image_array: 图像array数据
    :return:list 调整后的角度值
    """
    contours = contours_acquire(image_array)

    # 1、计算倾斜角度
    angle = cv2.minAreaRect(contours)[2]

    # 2、调整计算角度值
    def angle_process(angle):
        # 调整规则如下：
        """
            （-45,0]   之间，则为左倾：x => x
            （-90，-45]之间，则为右倾：x => x+90

            最终得到的角度：
                为负的表示左倾
                为正的表示右倾
        """
        if float(angle) < -45.:
            angle = float(angle) + 90
        else:
            angle = float(angle)

        return round(angle, 3)

    angle = angle_process(angle)

    return angle

