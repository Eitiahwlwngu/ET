'''
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import class_mapper
from flask_cors import CORS
from flask import Flask,make_response,session
from datetime import timedelta



import os

app = Flask(__name__)



app.config['SECRET_KEY'] = os.urandom(24)  # 随即产生24个字节的字符串


# 这个问题好坑，搞了好久，SameSite有三个值，
# Strict 完全禁止第三方 Cookie，跨站点时，任何情况下都不会发送 Cookie
# Lax 允许部分第三方请求携带 Cookie （默认）
# None 无论是否跨站都会发送 Cookie

# app.config["SESSION_COOKIE_SECURE"] = False
# app.config["SESSION_COOKIE_SAMESITE"] = "None"


CORS(app, supports_credentials=True)


# 设置连接数据库的URL，以下连mysql
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:123456@localhost:3306/ykjb'
# 以下连sql server
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mssql://root:123456@www.csoftedu.net:3306/testdb'
# 以下连oracle
# app.config['SQLALCHEMY_DATABASE_URI'] = 'oracle://root:123456@www.csoftedu.net:3306/testdb'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)



# 对象转字典，用于select * 的情况
def as_dict_all(objs):
    return [dict((col.name, getattr(obj, col.name)) for col in class_mapper(obj.__class__).mapped_table.c) for obj in
            objs]

# 对象转字典，用于查询某些字段的情况
#def as_dict_part(objs):
#    for obj in objs:
#        o=obj.keys()

#    return [dict(zip(obj.keys(), obj)) for obj in objs]

# 对象转字典，用于查询某些字段的情况
def as_dict_part(objs):
    # 初始化一个空列表用于存储结果
    result = []
    for obj in objs:
        # 将obj的_mapping属性转换为字典
        obj_dict = dict(obj._mapping)
        # 将转换后的字典添加到结果列表中
        result.append(obj_dict)
    # 返回结果列表
    return result

'''
# src/db.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import class_mapper
from flask_cors import CORS
import os
from flask_bcrypt import Bcrypt

app = Flask(__name__)

app.config['SECRET_KEY'] = os.urandom(24)  # 随机生成24个字节的字符串

CORS(app, supports_credentials=True)

# 设置连接数据库的URL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:root@localhost:3306/ykjb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secure_key'  # 用于会话加密

# 初始化数据库和 Bcrypt
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)  # 正确初始化 Bcrypt（需在 app 上下文中）

#db = SQLAlchemy(app)

def as_dict_all(objs):
    return [dict((col.name, getattr(obj, col.name)) for col in class_mapper(obj.__class__).mapped_table.c) for obj in objs]

def as_dict_part(objs):
    result = []
    for obj in objs:
        obj_dict = dict(obj._mapping)
        result.append(obj_dict)
    return result