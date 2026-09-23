# 系统库导入
import os
import time
from hashlib import md5  # 如果需要使用哈希功能

# 数据库相关导入
from db import *
from models import *
from sqlalchemy import and_, or_, func
from pymysql import *

# Flask 相关导入
from flask import Blueprint, request, jsonify, session, make_response, current_app

# 第三方库导入
import requests.api
import requests.auth

#import geoip2.database

# 自定义模块导入
#from usersapi import *
from disease_recognition_service import DiseasePredictor  # 确保导入的是正确的服务



def get_user_info_api():# 获取用户信息接口
    try:
        # 从会话中获取jk_user
        jk_user = session.get("jk_user")
        # print(jk_user)
        # 判断jk_user是否存在
        if jk_user is not None:
            # 在users表中查询tel等于jk_user的记录
            user = users.query.filter(users.tel == jk_user).first()
            # 如果查询到了用户
            if user:
                # 返回用户信息
                return jsonify({"res": as_dict_all([user])})
            # 如果没有查询到用户
            else:
                # 返回错误信息
                return jsonify({"res": "error"})
        # 如果jk_user为空
        else:
            # 返回错误信息
            return jsonify({"res": "error"})
    # 捕获所有异常
    except Exception as e:
        # 返回错误信息
        return jsonify({"res": "error"})


def change_pwd_api(old_password, new_password):
    try:
        # 假设这是您的密码修改逻辑
        user = Users.query.filter_by(nickname=session.get('jk_user')).first()
        if not user:
            return {"success": False, "message": "用户未登录"}

        if not bcrypt.check_password_hash(user.pwd, old_password):
            return {"success": False, "message": "旧密码错误"}

        user.pwd = bcrypt.generate_password_hash(new_password).decode('utf-8')
        db.session.commit()
        return {"success": True, "message": "密码修改成功"}
    except Exception as e:
        return {"success": False, "message": str(e)}

def up_user_info_api(headpic, nickname, tel, email, introduce):
    try:
        # 假设这是您的用户信息更新逻辑
        user = Users.query.filter_by(nickname=session.get('jk_user')).first()
        if not user:
            return {"success": False, "message": "用户未登录"}

        user.nickname = nickname
        user.tel = tel
        user.email = email
        user.introduce = introduce
        if headpic:
            user.headpic = headpic

        db.session.commit()
        return {"success": True, "message": "用户信息更新成功"}
    except Exception as e:
        return {"success": False, "message": str(e)}
def get_user_page_infor_api(uid):# 获取用户个人主页信息接口
    try:
        # 查询用户信息
        user = users.query.filter(users.uid == uid).first()
        # 将用户信息转换为字典格式，并返回JSON响应
        return jsonify({"res": as_dict_all([user])})
    except Exception as e:
        # 捕获异常，并返回JSON响应
        return jsonify({"res": "error"})



def log_auth_attempt(user_id, status):
    """增强版日志记录函数"""
    try:
        # 获取终端指纹
        ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        ua = request.headers.get('User-Agent', 'Unknown')

        # IP地理位置解析（需GeoLite2数据库）
        with geoip2.database.Reader('GeoLite2-City.mmdb') as reader:
            geo = reader.city(ip)
            country = geo.country.name
            city = geo.city.name

            # 创建日志对象（扩展字段需修改表结构）
        log = LoginLog(
            user_id=user_id if user_id else 0,  # 失败时需处理无效ID
            login_status=status,
            ip_address=ip,
            user_agent=ua[:255],  # 截断超长UA
            # 扩展字段示例
            country=country,
            city=city
        )

        # 异步提交（需配置Celery）
        db.session.add(log)
        db.session.commit()

    except Exception as e:
        app.logger.error(f" 日志记录失败: {str(e)}")
        db.session.rollback()


#用户日志函数
def login():
    """用户日志管理接口"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = Users.query.filter_by(username=username).first()

    # 登录验证前记录尝试
    log_auth_attempt(user.id if user else 0, 'failure')

    if user and user.verify_password(password):
        # 验证成功后更新状态
        log = LoginLog.query.filter_by(user_id=user.id).order_by(LoginLog.login_time.desc()).first()
        if log:
            log.login_status = 'success'
            db.session.commit()

            # 生成访问令牌
        access_token = create_access_token(identity=user.id)
        return jsonify({
            'status': 'success',
            'token': access_token
        })

    return jsonify({'status': '认证失败'}), 401

#注册功能逻辑函数
def regapi(tel, pwd, confirm_pwd, nickname):
    try:
        # 校验手机号格式
        if len(tel) != 11 or not tel.isdigit():
            return jsonify({"res": "手机号格式不正确"}), 400

        # 校验两次密码是否一致
        if pwd != confirm_pwd:
            return jsonify({"res": "两次密码不一致"}), 400

        # 检查手机号或昵称是否已被注册
        existing_user_by_tel = Users.query.filter_by(tel=tel).first()
        existing_user_by_nickname = Users.query.filter_by(nickname=nickname).first()

        if existing_user_by_tel:
            return jsonify({"res": "该手机号已被注册"}), 400
        if existing_user_by_nickname:
            return jsonify({"res": "该用户名已被占用，请更改后重试"}), 400

        # 使用 bcrypt 加密密码
        hashed_pwd = bcrypt.generate_password_hash(pwd).decode('utf-8')
        # 使用 bcrypt 加密密码
       # hashed_pwd = bcrypt.hashpw(pwd.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # 创建新用户并保存
        new_user = Users(
            tel=tel,
            pwd=hashed_pwd,
            nickname=nickname,
            email="",  # 默认为空
        )
        db.session.add(new_user)
        db.session.commit()

        return jsonify({"res": "ok"}), 201  # 成功状态码

    except Exception as e:
        app.logger.error(f"注册失败: {str(e)}")
        db.session.rollback()
        return jsonify({"res": "系统错误"}), 500




