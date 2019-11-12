# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/11/12 16:46

lst = ["jpg", "JPG", "gif", "PNG"]
for i in lst:
    if i.lower() in ("jpg", "png", "jpeg"):
        print(i)