# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/11/7 10:54

"""
输入url，下载图像到本地
单个url下载，调用url_download_picture(url)
csv文件中多个url下载，调用url_download_picture_batch(file_path)
ps：file_path的url_list.csv中必须包含有“image_url”的一列
"""
# 调用通用包
import urllib.request
import pandas as pd
import os
import sys
import time

# 将项目文件夹添加为环境变量(否则无法导入以下自定义模块）
current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)

# 调用自定义模块
from lib.yjp_ml_log import log

def url_download_picture(record, picture_save_path = "../data/picture"):
    """
    根据url下载图片，并保存至指定文件夹
    :param record: 需要下载的图像url信息    record: ["productinfo_id", "image_id","image_format", "image_url"]
    :param picture_save_path: # 图片保存本地文件位置
    :return:None
    """
    url = str(record[-1])      # 防止image_url为空无法判断
    if url != "nan":
        image_name = str(int(record[0])) + "_" + str(int(record[1]))
        image_format = str(record[2]).replace(" ", "")
        # print(image_name, image_format)
        save_path = picture_save_path + "/" + str(image_name) + "." + str(image_format)
        urllib.request.urlretrieve(url, save_path) # 根据url下载图像，并保存为save_path
    else:
        # print("productinfo_id = " + str(int(record[0])) + " 的url为空")
        log.logger.info("productinfo_id = {} 的url为空".format(str(int(record[0]))))

def url_download_picture_batch(file_path, picture_save_path = "../data/picture"):
    """

    :param file_path:表头含有“url”字段的csv文件，url下为所有url地址
    :param picture_save_path:图像保存路径，默认为本地
    :return:
    """
    log.logger.info("-*-url_download_picture_batch-*-")

    def read_csv(file_path):
        import pandas as pd
        data = pd.read_csv(file_path, encoding="utf-8")
        url_array = data.loc[:, ["productinfo_id", "image_id","image_format", "image_url"]].values

        return url_array.tolist()

    # 1、从本地csv文件导入url相关信息
    url_list = read_csv(file_path)
    # print(url_list)
    # print(type(url_list))
    download_sum = len(url_list)

    # 2、开始循环批量下载
    i = 0
    time0 = time.time()
    for record in url_list:
        # record: ["productinfo_id", "image_id","image_format", "image_url"]

        # 1、判断image_url不为空
        url = str(record[-1])  # 防止image_url为空无法判断
        if url != "nan":

            # 2、判断该文件是否已经存在，不存在的再去下载
            image_name = str(int(record[0])) + "_" + str(int(record[1]))
            image_format = str(record[2]).replace(" ", "")
            # print(image_name, image_format)
            save_path = picture_save_path + "/" + str(image_name) + "." + str(image_format)
            if not os.path.exists(save_path):
                url_download_picture(record, picture_save_path)

        i += 1
        if i%50 == 0:
            time_length = time.time() - time0
            log.logger.info("图像已下载{}/{}个, 下载已耗时: {}s".format(i,download_sum,time_length))

    # print("url下载完毕，图像已保存至： ", picture_save_path)
    time_sum = time.time() - time0
    log.logger.info("url下载完毕，图像已保存至：{}, 下载总耗时： {}s".format(picture_save_path, time_sum))

if __name__ == '__main__':

    # url_list.csv中必须包含有“image_url”的一列表名
    file_path = "../data/url_list_20191112.csv"
    url_download_picture_batch(file_path)

