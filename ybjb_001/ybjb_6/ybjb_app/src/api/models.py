# src/api/models.py
from ybjb_app.src.api.db import *  # 假设你的工作目录是 ybjb_app/

class Users(db.Model):
    __tablename__ = 'users'
    __table_args__ = {'extend_existing': True}

    uid = db.Column(db.Integer, primary_key=True)  # 用户ID，主键
    tel = db.Column(db.String(11))  # 电话号码
    pwd = db.Column(db.String(225))  # 密码 # 自我介绍
    nickname = db.Column(db.String(50))  # 昵称  # 头像链接
    email = db.Column(db.String(50))

    def __repr__(self):
        return f"<Users {self.nickname}>"

'''
    uid = db.Column(db.Integer, primary_key=True)
    tel = db.Column(db.String(50))
    pwd = db.Column(db.String(50))
    introduce = db.Column(db.String(200))
    nickname = db.Column(db.String(50))
    headpic = db.Column(db.String(50))
    email = db.Column(db.String(50))
'''

class Prediction(db.Model):
    __tablename__ = 'predictions'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(80))
    left_eye_image = db.Column(db.String(120))
    right_eye_image = db.Column(db.String(120))
    disease_type = db.Column(db.String(80))
    probability = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime(), default=db.func.current_timestamp())

class LoginLog(db.Model):
    __tablename__ = 'LoginLog'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    login_status = db.Column(db.String(50))
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(255))
    country = db.Column(db.String(100))
    city = db.Column(db.String(100))
    login_time = db.Column(db.DateTime, default=db.func.current_timestamp())






#
# # models.py
# from flask_sqlalchemy import SQLAlchemy
#
# db = SQLAlchemy()
#
# class users(db.Model):
#     __tablename__ = 'users'  # 表名
#
#     uid = db.Column(db.Integer, primary_key=True)  # 用户ID，主键
#     tel = db.Column(db.String(50))  # 电话号码
#     pwd = db.Column(db.String(50))  # 密码
#     introduce = db.Column(db.String(200))  # 自我介绍
#     nickname = db.Column(db.String(50))  # 昵称
#     headpic = db.Column(db.String(50))  # 头像链接
#     email = db.Column(db.String(50))  # 邮箱
#
#
# class articles(db.Model):
#     __tablename__ = 'articles'
#
#     aid = db.Column(db.Integer, primary_key=True)
#     title = db.Column(db.String(200))
#     con = db.Column(db.Text)
#     pubtime = db.Column(db.DateTime)
#     uid = db.Column(db.Integer)
#     viewcount = db.Column(db.Integer)
#     comcount = db.Column(db.Integer)
#     cid = db.Column(db.Integer)
#     haveimg = db.Column(db.Boolean)
#     imgsrc = db.Column(db.String(100))
#
#
# class category(db.Model):
#     __tablename__ = 'category'
#     cid = db.Column(db.Integer, primary_key=True)
#     cname = db.Column(db.String(200))
#
#
# class comments(db.Model):
#     __tablename__ = 'comments'
#     comid = db.Column(db.Integer, primary_key=True)
#     aid = db.Column(db.Integer)
#     uid = db.Column(db.Integer)
#     con = db.Column(db.Text)
#     pubtime = db.Column(db.DateTime)