# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/11/7 11:51

"""
程序功能：
    1、下载url：      根据sql下载mysql中的image_url数据，保存到"../data/url_list_{datetime}.csv"
    2、下载图像：     提取数据中的image_url，下载图像，保存到"../data/picture"
    3、图像分类预测：  对下载到"../data/picture"中的所有图像进行预测分类
    4、保存预测结果：  将预测结果保存到"../data/prediction_result_{datetime}.csv"文件(注意此处保存路径)

input:sql_path
output:image_url, picture, prediction_result
return:None
"""
# 调用通用模块
import os
import datetime
import csv
import pandas as pd
import time
import sys

# 将项目文件夹添加为环境变量(否则无法导入以下自定义模块）
current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)

# 调用自定义模块
import lib.mysql_process as local_mp
import lib.url_download_picture as local_udp
import bin.model_prediction as local_mpred
import lib.url_label_concat as local_ulc
from lib.yjp_ml_log import log

def main_process(sql_path = "../data/sql.txt", section = "MYSQL_TRADING_DEV"):
    log.logger.info("-*-main_process-*-")
    time0 = time.time()

    # # 1、根据sql下载mysql中的image_url数据，保存到"../data/url_list_{datetime}.csv"
    # local_mp.mysql_process(sql_path, section)
    # # print("sql提取完毕！")
    time1 = time.time()
    # log.logger.info("sql提取数据耗时：{}s".format(round(time1-time0, 3)))
    #
    # # # 2、提取数据中的image_url，下载图像，保存到"../data/picture"
    # import datetime
    url_file_path = "../data/url_list_" + str(datetime.date.today().strftime("%Y%m%d")) + ".csv"
    # local_udp.url_download_picture_batch(url_file_path)
    # # print("图像下载完毕！")
    time2 = time.time()
    # log.logger.info("图像下载耗时：{}s".format(round((time2 - time1), 3)))

    # 3、对下载到"../data/picture"中的所有图像进行预测分类
    path = "../data/picture"
    res = local_mpred.model_prediction(path)
    # print(res[:5])
    # print(type(res))
    # print("图像预测分类完毕！")
    time3 = time.time()
    log.logger.info("图像分类耗时：{}s".format(round(time3 - time2, 3)))


    # 4、将预测结果label与url信息进行匹配
    res_with_url = local_ulc.url_label_concat(res, url_file_path)

    # 5、将预测结果保存到"../data/prediction_result_{datetime}.txt"文件
    import csv
    prediction_result_save_path = "../data/prediction_result_" + str(datetime.date.today().strftime("%Y%m%d")) + ".txt"
    # print(prediction_result_save_path)

    with open(prediction_result_save_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        # 创建保存预测数据的csv文件
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(res_with_url.columns.tolist())
        for i in range(len(res_with_url)):
            csv_writer.writerow(res_with_url.iloc[i].values.tolist())
    # print("预测数据保存完毕！保存地址： ", prediction_result_save_path)
    time4 = time.time()
    log.logger.info("预测结果保存耗时：{}s".format(round(time4 - time3, 3)))
    log.logger.info("预测数据保存地址：{}".format(prediction_result_save_path))
    log.logger.info("程序总耗时：{}s".format(round((time4 - time0), 3)))


if __name__ == '__main__':
    log.logger.info("-*- 程序运行开始 -*-")

    # 1、读取系统默认输入参数
    import sys
    # 1.定义变量并初始化, 参数错误退出脚本
    try:
        log.logger.info(sys.argv)
        if len(sys.argv) == 1:
            sql_path = "../data/sql.txt"
            section = "MYSQL_TRADING_DEV"  # mysql默认配置环境信息
        elif len(sys.argv) == 2:
            sql_path = sys.argv[1]
            section  = "MYSQL_TRADING_DEV"  # mysql默认配置环境信息
        elif len(sys.argv) == 3:
            sql_path = sys.argv[1]
            section  = sys.argv[2]
        else:
            # print("---------the arguments you have inputed was wrong, please checks the arguments' numbers!-----------")
            log.logger.info("The arguments you have inputed was wrong!")
            sys.exit(1)
    except:
        # print("---------the arguments you have inputed was wrong, please checks the argument's type!-----------")
        log.logger.info("The arguments you have inputed was wrong!")
        sys.exit(1)

    main_process(sql_path, section)
