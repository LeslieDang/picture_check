# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/11/11 10:28

import lib.mysql_process as local_mp
import lib.url_download_picture as local_udp
import bin.model_prediction as local_mpred

import os
import datetime
import csv
import pandas as pd

def url_label_concat(res, url_file_path = "../data/url_list_" + str(datetime.date.today().strftime("%Y%m%d")) + ".csv"):
    """
    将预测得到的label结果，与url文件中的url信息进行匹配合并
    :param res: model_prediction返回得到的结果
    :param url_file_path: url文件路径
    :return:
    """

    # 一、读取url文件
    def read_csv(file_path):
        import pandas as pd
        data = pd.read_csv(file_path, encoding="utf-8")
        url_df = data.loc[:, ["productinfo_id", "image_id", "image_format", "image_url"]]

        return url_df

    url_df = read_csv(url_file_path)
    # print(url_df[:5])
    # print(type(url_df.iloc[0,1]))
    # print(url_df.iloc[0,1])

    # 二、对res得到的label数据进行处理
    # 1、将image_name中的"productinfo_id", "image_id"拆分开
    res_new = []
    res_new.append(["image_id", 'label', 'proba', "create_time"])
    create_time = str(datetime.date.today())
    for line in res[1:]:
        line_new = []
        image_id = line[0].split("_")[-1]
        line_new.append(int(image_id))
        line_new.append(int(line[2]))
        line_new.append(float(line[-1][-1]))
        line_new.append(create_time)
        # print(id)
        res_new.append(line_new)
    # print(res_new)

    # 将list转为DataFrame
    res_df = pd.DataFrame(res_new[1:], columns=res_new[0])
    # print(res_df)
    # print(type(res_df.iloc[0,1]))
    # print(res_df.iloc[0,1])

    # 三、将url与label进行匹配汇总
    out_put = pd.merge(res_df, url_df, on="image_id", how="left")
    # print(out_put)

    return out_put


if __name__ == '__main__':
    url_file_path = "../data/url_list_" + str(datetime.date.today().strftime("%Y%m%d")) + ".csv"

    res = [['image_name', 'image_format', 'label', 'proba'], ['100_14693', 'jpg', 0, [0.95, 0.05]],
              ['101_14695', 'jpg', 0, [0.95, 0.05]]]

    res_url = url_label_concat(res, url_file_path)
    print(res_url)
    for i in range(len(res_url)):
        print(res_url.iloc[i].values.tolist())

    print(res_url.columns.tolist())

    # prediction_result_save_path = "../data/prediction_result_" + str(datetime.date.today().strftime("%Y%m%d")) + ".txt"
    # res_url.to_csv(prediction_result_save_path,sep = ",", index = False)


