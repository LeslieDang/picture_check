# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/11/6 10:07


"""
function：根据本地图像路径，返回图像特征list(list[0]为特征表头)
input： image_path，image_path可以是一张图的路径，也可以是图片文件夹的路径
output: image_features，第一行为特征表头
ps：由于获取文件内存大小的特征需要读取源文件，因此，此处的输入须为图像的路径地址，而不是图像的image_array

图像特征获取
    主要提取的特征有：
    1、图像名：str
    2、图像文件格式：str
    3、图像文件大小：int
    4、图像尺寸：tuple (w, h)
    5、图像内物体位置、尺寸特征：tuple (x, y, w, h)
        物体外切矩形左上角位置：（x, y)
        物体外切矩形尺寸：(w, h)
    6、获取图像内物体的倾斜角度(图像轮廓的倾斜角度)：float
    7、图像四边留白百分比：tuple (top/h, down/h, left/w, right/w)
    8、图像内物体图幅占比：float
    9、物体是否居中，偏离中心不超过5%算居中：tuple (deviation, is_deviation)
    10、图像清晰度：int （值越大表示图像越清晰，反之表示越模糊）
    11、图像亮度计算：float （HSV中，object部分的V的均值作为图像亮度值）
    12、图像对比度计算:float
    13、白色像素的占比：用于判断背景色: float
    14、最小外切矩形外的灰度图像的像素的平均颜色：用于判断背景色: int


尽量统一接口：特征获取的输入均为image_array
统一输出：一个输出结果
"""
# 调用通用包
import cv2
import math
import time
import os
import sys


# 将项目文件夹添加为环境变量(否则无法导入以下自定义模块）
current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)

# 调用本地包
import lib.incline_angle as ia
from lib.yjp_ml_log import log

# 4、图像尺寸：tuple (w, h)
def image_size(image_array):
    h, w = image_array.shape[:2]
    return (w, h)

# 5、图像内物体位置、尺寸特征：tuple (x, y, w, h)
def thresh_process(image_array):
    """
    辅助函数：对图像进行二值化处理（转为黑白图）
        进行灰度处理
        二值化为黑白图像
    :param image_array: array 图像矩阵
    :return: 返回黑白图像tresh
    """

    # 将图片转化为灰度图
    img_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # 将图像二值化为黑白图片，1表示大于阈值的变为0，否则变为最大值255
    ret, thresh = cv2.threshold(img_gray, 127, 255, 1)  # (输入的灰度图像，阈值，最大值，划分时使用的算法)

    return thresh

def object_location(image_array):
    thresh = thresh_process(image_array)
    x, y, w, h = cv2.boundingRect(thresh)
    return (x, y, w, h)

# 6、获取图像内物体的倾斜角度(图像轮廓的倾斜角度)：float
def incline_angle(image_array):
    angle = ia.calculate_angle(image_array)
    angle = abs(angle)
    if angle > 45.:
        return round(90-angle, 3)
    else:
        return round(angle, 3)

# 7、图像四边留白百分比：tuple (top/h, down/h, left/w, right/w)
def white_space(image_array):
    image_size_ex = image_size(image_array)
    image_size_in = object_location(image_array)

    top = image_size_in[1]
    down = image_size_ex[1] - image_size_in[-1] - image_size_in[1]
    left = image_size_in[0]
    right = image_size_ex[0] - image_size_in[2] - image_size_in[0]

    return (top, down, left, right)

def white_space_percent(image_array):
    #(top/h, down/h, left/w, right/w)
    size = image_size(image_array)                  # 图像尺寸（w, h）
    space = white_space(image_array)

    top = space[0]/size[1]
    down = space[1]/size[1]
    left = space[2]/size[0]
    right = space[3]/size[0]

    return (round(top, 3), round(down, 3), round(left, 3) , round(right, 3))

# 8、图像内物体图幅占比：float
def object_percent(image_array):
    size = image_size(image_array)                  # 图像尺寸（w, h）
    image_size_in = object_location(image_array)    # 内部物体位置(x, y, w, h)

    return round((image_size_in[2] * image_size_in[3]) / (size[0] * size[1]), 3)

# 9、物体是否居中，偏离中心不超过5%算居中：tuple (deviation, is_deviation)
def is_center(image_array):
    size = image_size(image_array)                  # 图像尺寸（w, h）
    image_size_in = object_location(image_array)    # 内部物体位置(x, y, w, h)

    size_center = (size[0]/2, size[1]/2)            # 图像中心位置
    size_in_center = (image_size_in[0] + image_size_in[2]/2, image_size_in[1] + image_size_in[3]/2) # 内部物体中心位置

    deviation_h = (size_in_center[0] - size_center[0]) / size[0] # 水平偏离度
    deviation_v = (size_in_center[1] - size_center[1]) / size[1] # 垂直偏离度

    deviation = round(math.sqrt(deviation_h * deviation_h + deviation_v * deviation_v), 3)

    # 偏离中心点5%，即认为偏心。
    if deviation < 0.05:
        is_deviation = False
    else:
        is_deviation = True

    return (deviation, is_deviation)

# 10、图像清晰度：int（值越大表示图像越清晰，反之表示越模糊）
def get_image_var(image_array):
    img2gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    image_var = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return int(image_var)

# 11、图像亮度计算（HSV中，object部分的V的均值作为图像亮度值）
def get_image_brightness(image_array):

    # 1、将图像由RGB转为HSV
    image_hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    # print(image_hsv.shape)
    # print(len(image_hsv))
    # print(len(image_hsv[0]))
    # print(image_hsv[0][0])
    # 计算平均亮度
    w, h = image_hsv.shape[:2]
    brightness_sum = 0
    for i in range(w):
        for j in range(h):
            brightness_sum += image_hsv[i][j][2]
    brightness_avg = brightness_sum / (w * h)
    # print(brightness_avg)

    # 考虑图幅，边缘部分均为白色亮边
    percentage = object_percent(image_array)
    brightness_avg_object =  (brightness_avg - 255*(1-percentage))/percentage

    return round(brightness_avg_object, 3)

# 12、图像对比度计算:float
def get_contrast(image_array):
    # 四近邻对比度计算
    # 1、彩色转为灰度图片
    img_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    m, n = img_gray.shape
    # print("img_gray = \n", img_gray)
    # print(img_gray.shape)

    # 2、图片矩阵向外扩展一个像素
    img_gray_ext = cv2.copyMakeBorder(img_gray,1,1,1,1,cv2.BORDER_REPLICATE)
    rows_ext,cols_ext = img_gray_ext.shape
    # 3、计算对比度
    b = 0.0
    for i in range(1,rows_ext-1):
        for j in range(1,cols_ext-1):
            b += ((int(img_gray_ext[i,j])-int(img_gray_ext[i,j+1]))**2
                + (int(img_gray_ext[i,j])-int(img_gray_ext[i,j-1]))**2
                + (int(img_gray_ext[i,j])-int(img_gray_ext[i+1,j]))**2
                + (int(img_gray_ext[i,j])-int(img_gray_ext[i-1,j]))**2)
    contrast = b/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4)

    return round(contrast, 3)

# 13、白色像素的占比：用于判断背景色: float
def blank_percent(image_array):
    shape_ex = image_array.shape # 行*列
    h = shape_ex[0]
    w = shape_ex[1]
    count_blank = 0
    for i in range(h):
        for j in range(w):
            if image_array[i][j].tolist()[:3] == [255, 255, 255]:
                count_blank += 1
    blank_percentage = round(count_blank / w / h, 3)

    return blank_percentage

# 14、最小外切矩形外的灰度图像的像素的平均颜色：用于判断背景色: int
# 注意：img_gray.shape：列*行！！！
def blank_background(image_array):
    h, w = image_array.shape[:2]
    shape_ex = (w, h)
    # print("shape_ex = ", shape_ex)
    img_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    # 将灰度图转化为黑白图  threshold：将图像二值化为黑白图片
    ret, thresh = cv2.threshold(img_gray, 180, 255, 1)  # 1 表示将127-255的部分变为0（黑）,0-127的部分变为255（白）
    shape_in = cv2.boundingRect(thresh)
    # print("shape_in = ", shape_in)

    # 1、先判断点是否在外切矩形外
    # 2、将在外的像素点计算其灰度图的值
    if shape_ex == shape_in[2:4]:
        return 0

    gray_sum = 0
    count_pixel = 0
    for i in range(shape_ex[1]):
        for j in range(shape_ex[0]):
            if not (shape_in[0] <= j \
                    and j <= shape_in[0] + shape_in[2] \
                    and shape_in[1] <= i \
                    and i <= shape_in[1] + shape_in[3]):
                gray_sum += img_gray[i, j]
                count_pixel += 1
                # print(i,j)

    if gray_sum != 0:
        background_color = int(gray_sum / count_pixel)
    else:
        background_color = 0  # 0代表黑色

    return background_color



# 获取以上所有图像特征的函数
def picture_feature_extract_single(picture_path):
    # 1、图像名：str
    # 2、图像文件格式：str
    image_name, image_format = picture_path.replace(" ", "").split("/")[-1].split("\\")[-1].split(".")

    # 3、图像文件大小：int 单位：KB
    file_size = int(int(os.path.getsize(picture_path))/1000)  # 获取图像文件的内存大小，单位Byte（输入文件时，单位改为KB）

    # cv2.IMREAD_UNCHANGED： 用于读入图像的alpha通道，以保证位深度不变
    image_array = cv2.imread(picture_path, cv2.IMREAD_UNCHANGED)

    # 4、图像外尺寸 tuple(w, h)
    size = image_size(image_array)

    # 5、图像内物体位置、尺寸特征：tuple (x, y, w, h)
    location = object_location(image_array)

    # 6、获取图像内物体的倾斜角度(图像轮廓的倾斜角度)：float
    angle = incline_angle(image_array)

    # 7、图像四边留白百分比：tuple (top/h, down/h, left/w, right/w)
    white_percent = white_space_percent(image_array)

    # 8、图像内物体图幅占比：float
    object_percentage = object_percent(image_array)

    # 9、物体是否居中，偏离中心不超过5%算居中：tuple (deviation, is_deviation)
    is_deviation = is_center(image_array)

    # 10、图像清晰度：int（值越大表示图像越清晰，反之表示越模糊）
    blur_value = get_image_var(image_array)

    # 11、图像亮度计算（HSV中，object部分的V的均值作为图像亮度值）
    image_brightness = get_image_brightness(image_array)

    # 12、图像对比度计算
    contrast = get_contrast(image_array)

    # 13、白色像素的占比：用于判断背景色
    blank_percentage = blank_percent(image_array)

    # 14、最小外切矩形外的灰度图像的像素的平均颜色：用于判断背景色
    blank_background_color = blank_background(image_array)

    return [image_name, image_format, file_size, size, location, angle, white_percent, object_percentage, is_deviation, blur_value
        , image_brightness, contrast, blank_percentage, blank_background_color]

def file_path_and_name_list(file_path):
    """
    # 1、批量获取文件路径
    :param file_path: 图片文件夹的路径，绝对路径获相对路径均可
    :return: 图片文件夹的绝对路径，图片文件名列表list
    """
    import os

    os.chdir(file_path)
    a = os.getcwd()  # 获取完整路径

    for root, dirs, files in os.walk(a, topdown=False):
        # root： # str  根目录
        # dirs： # list 根目录下的子文件夹
        # files：# list 文件夹下所有文件名集合
        file_path = root
        image_name_list = files

    return (file_path, image_name_list)

def picture_feature_extract(path):
    time0 = time.time()
    # 文件/文件夹判断
    import os
    if os.path.isfile(path):
        # 1、一个图像的特征提取
        header = ["image_name","image_format","file_size","size","location","angle","white_percent","object_percentage"
                 ,"is_deviation","blur_value","image_brightness","contrast","blank_percentage","blank_background_color"]
        features_list = []
        features_list.append(header)

        features = picture_feature_extract_single(path)
        features_list.append(features)

        return features_list
        # print(features_list)
        # print(type(features_list))
    elif os.path.isdir(path):
        # 2、一个文件夹下所有图像的特征提取
        folder_path, image_name_list = file_path_and_name_list(path)
        picture_sum = len(image_name_list)
        # 循环提取每一个图像的特征
        header = ["image_name", "image_format", "file_size", "size", "location", "angle", "white_percent",
                  "object_percentage", "is_deviation", "blur_value", "image_brightness", "contrast",
                  "blank_percentage", "blank_background_color"]
        features_list = []
        features_list.append(header)

        i = 0
        for image_name_full in image_name_list:
            picture_path = folder_path + "/" + image_name_full
            features_list.append(picture_feature_extract_single(picture_path))

            i += 1
            if i % 5 == 0:
                time_length = time.time() - time0
                log.logger.info("图像特征已提取{}/{}个, 已耗时: {}s".format(i, picture_sum, time_length))

        time_sum = time.time() - time0
        log.logger.info("图像特征抽取完毕！总耗时{}s".format(time_sum))

        return features_list
    else:
        # print(path,"：不是一个有效路径")
        log.logger.info("{}：不是一个有效路径".format(path))




if __name__ == '__main__':

    path = "../data/picture/logo1.png"
    path = "../data/picture"
    features = picture_feature_extract(path)
    # print(features)
    # print(type(features))
    # print(len(features))