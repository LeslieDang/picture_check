# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/11/8 11:14

# 辅助函数：用于读取文件夹下的所有文件
def file_path_and_name_list(file_path):
    """
    # 1、批量获取文件夹下所有文件的路径
    :param file_path: 图片文件夹的路径，绝对路径获相对路径均可
    :return: 图片文件夹的绝对路径，图片文件名列表list
    """
    import os
    path_original = os.getcwd()
    print("切换前的工作路径 = \n",path_original)  # 获取完整路径
    os.chdir(file_path)
    a = os.getcwd()  # 获取完整路径
    print("切换后的工作路径 = \n", os.getcwd())  # 获取完整路径

    #
    # for root, dirs, files in os.walk(a, topdown=False):
    #     # root： # str  根目录
    #     # dirs： # list 根目录下的子文件夹
    #     # files：# list 文件夹下所有文件名集合
    #     file_path = root
    #     image_name_list = files


    os.chdir(path_original)
    print("切换后的工作路径 = \n", os.getcwd())  # 获取完整路径

    # return (file_path, image_name_list)

if __name__ == '__main__':
    path  = "../data/picture"
    file_path_and_name_list(path)