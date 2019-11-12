#!/usr/bin/env python
# -*- coding:utf-8 -*-


import configparser
import pymysql
import os
import sys

# 将项目文件夹添加为环境变量(否则无法导入以下自定义模块）
current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)

from lib.yjp_ml_log import log

def get_mysql_db(section):
    # 1、读取数据库配置文件信息
    config = configparser.ConfigParser()
    config.read("../config/properties.ini", encoding="utf-8")

    # 判断是否存在该配置项
    if not config.has_section(section):
        log.logger.info("../config/properties.ini 中不存在该section配置信息！")
        return
    else:
        # 打开数据库连接
        db = pymysql.connect(
             host    = config.get(section, "host")
            ,port    = int(config.get(section, "port"))
            ,user    = config.get(section, "user")
            ,passwd  = config.get(section, "passwd")
            ,db      = config.get(section, "db")
            ,charset = config.get(section, "charset")
        )
        return db

def get_data_mysql(sql, section):
    """
    根据SQL语句获取数据、表结构
    :param sql:
    :return: tuple数据、tuple 表头
    """
    db = get_mysql_db(section)
    cursor = db.cursor()  # 使用cursor()方法获取操作游标
    # SQL 查询语句
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()  # 获取数据
        fields = cursor.description  # 获取表结构
        # 提取表头
        head = []
        for field in fields:
            head.append(field[0])
        head = tuple(head)

        cursor.close()  # 关闭游标
        db.close()  # 关闭数据库连接
        return results, head
    except Exception as e:
        print('str(Exception):\t', str(Exception))
        print('str(e):\t\t', str(e))

    cursor.close()  # 关闭游标
    db.close()      # 关闭数据库连接

def save_csv(data, head = None, path = "../data/url_.csv"):
    """
    保存数据到csv文件，默认保存地址为当前文件夹
    :param data: 需要保存的数据
    :param head: 需要保存的数据的表头
    :param path: 保存路径
    :return: None
    """
    import csv
    with open(path, "w", newline="",encoding="utf-8-sig") as csv_file:
        # encoding="utf-8-sig"  文件存取时，要注意编码格式（此处不能使用encoding="utf-8"）
        csv_writer = csv.writer(csv_file)
        if head != None:
            csv_writer.writerow(head)
        for line in data:
            csv_writer.writerow(line)

def mysql_process(sql_path = "../data/sql.txt", section = "MYSQL_TRADING_DEV"):
    log.logger.info("-*-mysql_process-*-")
    sql = open(sql_path, "r", encoding="gbk").read()
    # print(sql)
    # print(type(sql))

    results, head = get_data_mysql(sql, section)  # 发返回的是tuple类型数据
    import datetime
    url_save_path = "../data/url_list_" + str(datetime.date.today().strftime("%Y%m%d")) + ".csv"
    save_csv(results, head, path=url_save_path)
    # print("sql提取的url文件已保存至： ", url_save_path)
    log.logger.info("sql提取的url文件已保存至： {}".format(url_save_path))

    # print(results[0], "\n", head)


if __name__ == '__main__':

    sql_path = "../data/sql.txt"
    section = "MYSQL_TRADING_DEV"
    mysql_process(sql_path, section)

