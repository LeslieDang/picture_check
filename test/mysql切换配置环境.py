# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/11/11 15:46



import configparser

def read_section(section):
    # 1、读取数据库配置文件信息
    config = configparser.ConfigParser()
    config.read("../config/properties.ini", encoding="utf-8")

    # 判断是否存在该配置项
    if not config.has_section(section):
        print("../config/properties.ini 中不存在该section配置信息！")
        return
    else:
        # 打开数据库连接
        host=config.get(section, "host")
        port=config.get(section, "port")
        user=config.get(section, "user")
        passwd=config.get(section, "passwd")
        db=config.get(section, "db")
        charset=config.get(section, "charset")
        print(charset)
        print(type(charset))
        print(len(charset))


if __name__ == '__main__':
    # 2、指定section名
    section = "MYSQL_TRADING_DEV"
    conf = read_section(section)