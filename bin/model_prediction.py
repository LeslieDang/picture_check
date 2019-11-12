# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/11/6 15:19

"""
主调用函数：model_prediction(path)

根据图像路径，返回预测分类，及分类概率
:param path: 图像路径(单个图像文件路径，或单个图像文件夹路径）
:return: [header，result1, result2...]
header = ["image_name", "image_format", "pre", "label"]
"""

# 通用模块
import datetime
import time
import joblib
import numpy as np
import csv
import os
import sys

# 将项目文件夹添加为环境变量(否则无法导入以下自定义模块）
current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)

# 自定义模块
import lib.picture_feature_extract as local_pfe
import lib.data_preprocessing as local_dp
import bin.model_train as local_model_train
from lib.yjp_ml_log import log

# 辅助函数：用于读取文件夹下的所有文件
def file_path_and_name_list(file_path):
    """
    # 1、批量获取文件夹下所有文件的路径
    :param file_path: 图片文件夹的路径，绝对路径获相对路径均可
    :return: 图片文件夹的绝对路径，图片文件名列表list
    """
    import os
    path_original = os.getcwd() # 获取当前工作路径，为后面修正做准备
    # print("初始的工作路径", os.getcwd())

    os.chdir(file_path)         # 更改工作路径到picture下
    a = os.getcwd()             # 获取完整路径
    # print("修改后的工作路径", os.getcwd())

    for root, dirs, files in os.walk(a, topdown=False):
        # root： # str  根目录
        # dirs： # list 根目录下的子文件夹
        # files：# list 文件夹下所有文件名集合
        file_path = root
        image_name_list = files

    os.chdir(path_original)     # 修正更改了得工作路径，恢复原样
    # print("恢复之后的工作路径", os.getcwd())

    return (file_path, image_name_list)

# 将图像的特征数据写入csv文件
def features_save(features_list):
    # print("features_list: ",features_list)
    import csv
    # <数据保存>将原始的特征数据写入本地文件夹，以便后期加入训练模型时使用
    with open("../data/feature_labeled.csv", "a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file) # 创建写入指针
        for features in features_list:
            # print("features : ", features)
            # 将每一行特征数据，写入csv文件（包含创建世界、预测正例概率、预测分类）
            writer.writerow(features)
    # print("新的图像的特征、预测数据已追加至： ../data/feature_labeled.csv 文件")
    log.logger.info("预测的图像的特征及预测数据已追加至： ../data/feature_labeled.csv 文件")

# 单个图像预测
def single_predict(path, scaler_reload, clf_reload):
    # 2、获取图像预测结果
    # 2.1 load data
    features_raw = local_pfe.picture_feature_extract(path)[1]  # 此处必须为单个图像的picture_path
    # print(features_raw)
    features_raw_to2dim = np.array(features_raw).reshape(1, len(features_raw))

    # 2.2 data preprocess
    features_process = local_dp.data_pre_process(features_raw_to2dim)
    # print(features_process)

    # 2.3 data standard
    features_process = scaler_reload.transform(features_process)

    label = clf_reload.predict(features_process)
    proba = clf_reload.predict_proba(features_process)

    features_raw.append(str(datetime.date.today()))
    features_raw.append(proba[0].tolist()[1])
    features_raw.append(label[0])

    return (label[0], proba[0].tolist(), features_raw) # (int, list, list)


def model_prediction(path):
    """
    主调用函数：model_prediction(path)

    根据图像路径，返回预测分类，及分类概率
    :param path: 图像路径(单个图像文件路径，或单个图像文件夹路径）
    :return: [header，result1, result2...]
    header = ["image_name", "image_format", "pre", "label"]
    """
    log.logger.info("-*-model_prediction-*-")
    time0 = time.time()

    # 1、reload model
    import os
    # 如果训练模型不存在，则先跑模型训练
    model_path = "../data/model/scaler_and_clf_" + str(datetime.date.today().strftime("%Y%m%d")) + ".pkl"
    if not os.path.exists(model_path):
        log.logger.info("载入模型不存在，开始训练模型")
        local_model_train.model_training_main()

    model_reload = joblib.load(model_path)
    scaler_reload = model_reload[0]
    clf_reload = model_reload[1]

    if os.path.isfile(path):
        res_list = []
        header = ["image_name", "image_format", "label", "proba"]
        res_list.append(header)

        # 1、获取图像名、图像格式
        image_name, image_format = str(path).replace(" ", "").split("/")[-1].split("\\")[-1].split(".")
        # 2、获取图像分类结果、分类概率
        label, proba, features_raw = single_predict(path, scaler_reload, clf_reload)

        res = [image_name, image_format, label, proba]
        res_list.append(res)

        # <数据保存>将特征数据写入本地csv文件
        features_list = []
        features_list.append(features_raw)
        features_save(features_list)

        return res_list

    elif os.path.isdir(path):
        res_list = []
        header = ["image_name", "image_format", "label", "proba"]
        res_list.append(header)

        # 1、读取文件夹内所有文件
        folder_path, image_name_list = file_path_and_name_list(path)
        picture_sum = len(image_name_list)
        # 2、循环遍历每一个文件

        # < 数据保存 > 将特征数据写入本地csv文件
        features_list = []

        predicted_num = 0
        for image_name_full in image_name_list:
            image_path = folder_path + "/" + image_name_full

            # 1、获取图像名、图像格式
            image_name, image_format = str(image_path).replace(" ", "").split("/")[-1].split("\\")[-1].split(".")
            if str(image_format).lower() in ("jpg", "jpeg", "png"):
                # 2、获取图像分类结果、分类概率
                label, proba, features_raw = single_predict(image_path, scaler_reload, clf_reload)

                res = [image_name, image_format, label, proba]
                res_list.append(res)

                # < 数据保存 > 将特征数据写入本地csv文件
                features_list.append(features_raw)

            predicted_num += 1
            if predicted_num % 50 == 0:
                time1 = time.time() - time0
                log.logger.info("已预测图像数目：{}/{}, 预测图像分类已耗时：{}s".format(predicted_num, picture_sum, time1))

        # < 数据保存 > 将特征数据写入本地csv文件
        features_save(features_list)

        time_length = time.time() - time0
        log.logger.info("预测图像总数：{}个，预测总耗时：{}s".format(picture_sum, time_length))

        return res_list

    else:
        log.logger.info("输入路径无效！")


if __name__ == '__main__':
    path = "../data/picture"
    # path = "../data/picture/fenjiu.jpg"
    res = model_prediction(path)
    print(res)