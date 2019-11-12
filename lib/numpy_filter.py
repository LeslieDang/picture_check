# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/11/8 13:46

import numpy as np

def numpy_filter(feature_data, proba_threshold = 0.7):
    # 1、筛选重复值、预测概率处在（1-threshold，threshold）之间的项
    image_name = []
    feature_data_filter1 = []

    # 统计符合条件的正负样本数
    count_0 = 0
    count_1 = 0

    for line in feature_data:
        name = str(line[0]).replace(" ", "")
        if not name in image_name:
            if (1-proba_threshold) >= float(line[-2]) or float(line[-2]) >= proba_threshold:
                image_name.append(name)
                feature_data_filter1.append(line)
                if int(line[-1]) == 0:
                    count_0 += 1
                if int(line[-1]) == 1:
                    count_1 += 1

    # print("image_name = ", image_name)
    # print("feature_data_filter = \n", feature_data_filter1)

    # print("筛选后的正样本数 = ", count_1)
    # print("筛选后的负样本数 = ", count_0)

    # 2、将正负样本数调整到一样多
    num_filter = min(count_0, count_1)
    feature_data_filter2 = []

    # 统计已经选入的符合条件的正负样本数
    num_0 = num_filter
    num_1 = num_filter

    for line in feature_data_filter1:
        if int(line[-1]) == 0:
            if num_0 > 0:
                feature_data_filter2.append(line)
                num_0 -= 1
        elif int(line[-1]) == 1:
            if num_1 > 0:
                feature_data_filter2.append(line)
                num_1 -= 1

    feature_data_filter2 = np.array(feature_data_filter2)
    # print("feature_data_filter2 = \n", feature_data_filter2)
    return feature_data_filter2

if __name__ == '__main__':
    feature_data_train_raw = np.array([[1,"jpg",0,1],
                    [2,"jpg",1,1],
                    [3,"jpg",1,1],
                    [4,"jpg",0,1],
                    [5,"jpg",0.85,1],
                    [5,"jpg",0.75,1],
                    [6,"jpg",0.2 ,0],
                    [7,"jpg",0.5 ,0],
                    [8,"jpg",0.6 ,1],
                    [8,"jpg",0.75 ,1],
                    ["1 1","jpg",0.75 ,1],
                    [11,"jpg",0.75 ,1]])

    feature_data_filter = numpy_filter(feature_data_train_raw, 0.5)
    print("feature_data_filter = \n", feature_data_filter)

