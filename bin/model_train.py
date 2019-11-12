# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/11/5 15:13

# model training : RandomForestClassifier
"""
input: feature_labeled_data_path
output: (scaler,Classifier)     # scaler：标准化算子，后期预测数据进行标准化时，需要用到的算子
"""

import pandas as pd
from sklearn import preprocessing
import datetime
import time
import joblib
import os
import sys

# 将项目文件夹添加为环境变量(否则无法导入以下自定义模块）
current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)


# 导入本地自定义模块
import lib.data_preprocessing as local_dp
import lib.numpy_filter as local_nf
from lib.yjp_ml_log import log


# RandomForestClassifier分类器训练
def RandomForest_modeling(feature_data, labels):
    # 1、切分训练集、测试集、验证集
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.3)
    # print("y_train = \n", y_train)

    # 2、引入评价指标
    from sklearn.metrics import precision_score, recall_score, f1_score

    # 3、引入分类器
    from sklearn.ensemble import RandomForestClassifier  # 随机森林  bagging集成方法
    clf = RandomForestClassifier(n_estimators = 100)
    clf.fit(x_train, y_train)

    # 4、在测试集上进行预测——测试性能
    y_pre = clf.predict(x_test)
    # print("precision: ", precision_score(y_test, y_pre))
    # print("recall   : ", recall_score(y_test, y_pre))
    # print("f1_scores: ", f1_score(y_test, y_pre))

    log.logger.info("precision: {}".format(precision_score(y_test, y_pre)))
    log.logger.info("recall   : {}".format(recall_score(y_test, y_pre)))
    log.logger.info("f1_scores: {}".format(f1_score(y_test, y_pre)))


    # # 5、模型性能测试展示: learning_curve
    # from sklearn.model_selection import learning_curve
    # import matplotlib.pyplot as plt
    #
    # train_sizes, train_score, test_score = learning_curve(RandomForestClassifier(n_estimators = 100), feature_data, labels,
    #                                                       train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1], cv=10,
    #                                                       scoring='accuracy')
    #
    # train_error = 1 - np.mean(train_score, axis=1)
    # test_error = 1 - np.mean(test_score, axis=1)
    # plt.plot(train_sizes, train_error, 'o-', color='r', label='training')
    # plt.plot(train_sizes, test_error, 'o-', color='g', label='testing')
    # plt.legend(loc='best')
    # plt.xlabel('traing examples')
    # plt.ylabel('error')
    # plt.show()

    return clf


def model_training_main(feature_data_path = "../data/feature_labeled.csv"):
    log.logger.info("-*-model_training_main-*-")
    time0 = time.time()
    # 一、模型训练
    # 1、读取数据
    feature_data_train_raw = pd.read_csv(feature_data_path, encoding="utf-8").values
    # print(feature_data_train_raw[:2])

    # 2、数据倾斜的处理、数据筛选
    feature_data_train_filter = local_nf.numpy_filter(feature_data_train_raw)
    train_sample_sum = len(feature_data_train_filter)
    # print(feature_data_train_filter[:2])
    # print(""0feature_data_train_filter.shape)

    labels = feature_data_train_filter[:, -1]
    # 更改文件格式到list，筛选有标签的记录，去掉空置，并将标签映射为整型
    labels = list(map(lambda x: int(x), filter(lambda x: str(x) != "nan", labels)))
    # print(labels)
    # print(len(labels))

    # 2、对特征数据进行预处理
    feature_data_train_preprocessed = local_dp.data_pre_process(feature_data_train_filter)
    # print(feature_data_train_preprocessed[:5])
    # print(type(feature_data_train_preprocessed))

    # 3、数据标准化(注意参数传递）
    # 获取标准化算子（用于统一训练数据与预测数据的标准化尺度）
    scaler = preprocessing.StandardScaler().fit(feature_data_train_preprocessed)
    # print(scaler.mean_)
    # print(scaler.var_)    # 方差（注意，标准化用的是标准差）
    # 训练数据标准化
    feature_data_train_preprocessed_standard = scaler.transform(feature_data_train_preprocessed)
    # print(feature_data_train_preprocessed_standard[:5])

    # 4、模型训练
    # 1、多模型预训练，寻找合适的模型
    # modeling(feature_data_train_preprocessed_standard, labels)
    # 2、筛选后的模型进行训练
    clf = RandomForest_modeling(feature_data_train_preprocessed_standard, labels)
    time_length = time.time() - time0
    # print("-----------------------模型训练结束-------------------------")
    log.logger.info("模型训练样本数：{} ，模型训练总耗时：{}s".format(train_sample_sum, time_length))

    # 5、模型保存
    model = (scaler, clf)
    model_save_path = "../data/model/scaler_and_clf_" + str(datetime.date.today().strftime("%Y%m%d")) + ".pkl"
    joblib.dump(model, model_save_path)
    log.logger.info("训练的模型已保存至：{}".format(model_save_path))

    return (scaler, clf)        # scaler：数据标准化算子  # clf：训练好的分类器


if __name__ == '__main__':

    feature_labeled_data_path = "../data/feature_labeled.csv"
    model_training_main(feature_labeled_data_path)
