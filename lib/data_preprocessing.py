# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/11/6 11:40

"""
function: 数据预处理,不包含正则化
input: numpy.array
output: numpy.array
"""
"""
输入数据格式：
    [image_name, image_format,is_deviation, file_size, size, location, angle, white_percent, object_percentage, 
     blur_value, image_brightness, contrast, blank_percentage, blank_background_color]
输出数据：
    每一列均为数值型
    新增组合特征
    [picture_scale_processed, white_percent, blur_per_pixel, file_size_per_pixel, file_size, angle, object_percentage, 
    is_deviation, blur_value, image_brightness, contrast, blank_percentage, blank_background_color]
    
"""

import numpy as np

# 2、数据预处理
def data_pre_process(data):
    # 将数据转为单列的数字格式（不进行正则化）

    # 辅助函数
    def integer_data_process(data_unit):
        # 对整数型的字符串还原成np.array。如：(400, 400) -> [400, 400]
        data_unit = str(data_unit).replace(" ", "").replace("(", "").replace(")", "").split(",")
        data_unit = list(map(lambda x: int(x), data_unit))
        return data_unit

    def float_data_process(data_unit):
        # 对浮点型的字符串还原成np.array。如：(0.145, 0.092) -> [0.145, 0.092]
        data_unit = str(data_unit).replace(" ", "").replace("(", "").replace(")", "").split(",")
        data_unit = list(map(lambda x: round(float(x), 3), data_unit))
        return data_unit

    def mixed_data_process(data_unit):
        # 对整数型的字符串还原成np.array。如：(400, True) -> [400, 1]
        data_unit = str(data_unit).replace(" ", "").replace("(", "").replace(")", "").replace("'", "").split(",")

        def transfer(unit):
            if unit == "True":
                return 1
            elif unit == "False":
                return 0
            else:
                return round(float(unit), 3)
        data_unit = list(map(transfer, data_unit))

        return data_unit

    # 1、size特征处理
    size_processed = np.array(list(map(integer_data_process, data[:, 3])))
    # print("size_processed.shape : ", size_processed.shape)

    # 2、picture_scale特征处理：图像宽 * 高
    picture_scale_processed = np.array(list(map(lambda x: x[0]*x[1], size_processed)))
    picture_scale_processed = picture_scale_processed.reshape((len(picture_scale_processed), 1))
    # print("picture_scale_processed.shape : ", picture_scale_processed.shape)

    # 3、location特征处理
    location_processed = np.array(list(map(integer_data_process, data[:, 4])))
    # print(location_processed[:5])
    # print("location_processed.shape : ",location_processed.shape)

    # 4、white_percent特征处理
    white_percent_processed = np.array(list(map(float_data_process, data[:, 6])))
    # print(white_percent_processed[:5])
    # print("white_percent_processed.shape : ",white_percent_processed.shape)

    # 5、is_deviation特征处理
    is_deviation_processed = np.array(list(map(mixed_data_process, data[:, 8])))[:,0]
    is_deviation_processed = is_deviation_processed.reshape((len(is_deviation_processed), 1))
    # print(is_deviation_processed[:5])
    # print("is_deviation_processed.shape : ",is_deviation_processed.shape)

    # 6、增加组合特征
    # 单位像素的清晰度*10000
    blur = data[:, 9].reshape(len(data[:, 9]), 1)
    blur_per_pixel = blur * 10000 / picture_scale_processed
    blur_per_pixel = np.around(blur_per_pixel.astype(float), 3)

    # 单位像素的文件内存大小*10000
    file_size = data[:, 2].reshape(len(data[:, 2]), 1)
    file_size_per_pixel = file_size * 10000 / picture_scale_processed
    file_size_per_pixel = np.around(file_size_per_pixel.astype(float), 3)


    data_processed = np.hstack(
        (
         # size_processed,
         picture_scale_processed
         , white_percent_processed
         , is_deviation_processed
         , blur_per_pixel
         , file_size_per_pixel
         , data[:, (2, 5, 7, 9, 10, 11, 12, 13)]
         )
    )
    # print("data_processed.shape = ", data_processed.shape)
    # print("data_processed = \n", data_processed[:5])

    return data_processed

if __name__ == '__main__':
    # 1、数据导入
    import pandas as pd
    def read_csv(file_path):
        # 1、从本地csv文件导入url相关信息
        data = pd.read_csv(file_path, encoding="utf-8")
        header = data.columns.values
        data = data.values

        return [header, data]

    feature_data_path = "../data/feature_labeled.csv"
    header, feature_data_train_raw = read_csv(feature_data_path)

    print("feature_data_train_raw = \n", feature_data_train_raw[:5])
    feature_data_train = data_pre_process(feature_data_train_raw)
    print("feature_data_train = \n", feature_data_train[:5])