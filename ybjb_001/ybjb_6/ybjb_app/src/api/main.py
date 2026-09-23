from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from disease_recognition_service import DiseasePredictor  # 假设这是正确的路径
from ybjb_app.src.api.models import *
from ybjb_app.src.api.db import *
from ybjb_app.src.api.usersapi import *
from flask_bcrypt import Bcrypt
import os
from openai import OpenAI




# 初始化 OpenAI 客户端
client = OpenAI(
    base_url='https://ms-fc-78295036-78fe.api-inference.modelscope.cn/v1',
    api_key='77131395-bf33-433a-beea-fda19e04b071',  # ModelScope Token
)

#疾病智能助手
@app.route('/chat', methods=["POST","GET"])
def chat():
    try:
        # 获取前端发送的用户输入
        user_input = request.json.get('message')
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400

        # 调用模型生成回复
        response = client.chat.completions.create(
            model='unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF',  # ModelScope Model-Id
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': user_input}
            ],
            stream=False  # 不使用流式输出
        )

        # 提取模型回复内容
        reply = response.choices[0].message.content
        return jsonify({'response': reply})

    except Exception as e:
        # 捕获异常并返回错误信息
        return jsonify({'error': str(e)}), 500




#用户注册请求
@app.route('/register', methods=["GET", "POST"])
def register():
    try:
        # 获取请求中的 JSON 数据
        data = request.get_json()

        # 检查必要参数是否存在
        required_fields = ['tel', 'pwd', 'confirm_pwd', 'nickname']
        if not all(field in data for field in required_fields):
            return jsonify({"res": "参数缺失"}), 400

        # 提取参数
        tel = data.get('tel')
        pwd = data.get('pwd')
        confirm_pwd = data.get('confirm_pwd')
        nickname = data.get('nickname')

        # 调用注册接口函数
        return regapi(tel=tel, pwd=pwd, confirm_pwd=confirm_pwd, nickname=nickname)

    except Exception as e:
        app.logger.error(f"注册路由错误: {str(e)}")
        return jsonify({"res": "系统错误"}), 500



# 登录接口
@app.route('/login', methods=["POST"])
def loginapi():
    try:
        data = request.get_json()

        if not data or not data.get('username_or_tel') or not data.get('pwd'):
            return jsonify({"res": "error", "message": "缺少必要参数"}), 400

        user = Users.query.filter_by(tel=data['username_or_tel']).first()
        if not user:
            user = Users.query.filter_by(nickname=data['username_or_tel']).first()

        if not user:
            return jsonify({"res": "error", "message": "用户名或手机号未注册，请先注册"}), 404

        if not bcrypt.check_password_hash(user.pwd, data['pwd']):
            return jsonify({"res": "error", "message": "密码输入错误，请重新输入"}), 401

        session['jk_user'] = user.nickname if user.nickname else user.tel
        session['jk_token'] = f"{str(request.user_agent)}_{request.remote_addr}"

        # 返回包含 data 字段的响应
        return jsonify({
            "res": "ok",
            "message": "登录成功",
            "data": {
                "nickname": user.nickname
            },
            "token": session['jk_token']
        }), 200

    except Exception as e:
        app.logger.error(f"Error in loginapi: {str(e)}")
        return jsonify({"res": "error", "message": "服务器内部错误"}), 500
# 检查用户名是否存在
@app.route('/check-username', methods=["GET"])
def check_username():
    username = request.args.get('username')
    if not username:
        return jsonify({"exists": False})

    user = Users.query.filter_by(nickname=username).first()
    return jsonify({"exists": user is not None})

# 检查手机号是否存在
@app.route('/check-tel', methods=["GET"])
def check_tel():
    tel = request.args.get('tel')
    if not tel:
        return jsonify({"exists": False})

    user = Users.query.filter_by(tel=tel).first()
    return jsonify({"exists": user is not None})

# 获取验证码
@app.route('/verification-code', methods=["GET"])
def get_verification_code():
    # 生成随机验证码
    import random
    code = ''.join(random.choices('0123456789', k=6))
    return jsonify({"code": code})

@app.route("/selall",methods=["GET", "POST"])
def selall():
    user_all=Users.query.all()
    return jsonify({"res":as_dict_all(user_all)})





#在usersapi.py中定义的函数，用于获取用户信息
@app.route("/getuserinfo", methods=["GET", "POST"])
def getuserinfo():
    return get_user_info_api()


#在usersapi.py中定义的函数，用于更新用户个人信息接口
@app.route("/upuserinfo", methods=["POST"])
def upuserinfo():
    try:
        data = request.get_json()
        nickname = data.get("name")
        tel = data.get("phone")
        email = data.get("email")
        introduce = data.get("bio")
        headpic = data.get("avatar")

        # 调用后端逻辑处理用户信息更新
        result = up_user_info_api(headpic, nickname, tel, email, introduce)
        return jsonify({"success": True, "message": "用户信息更新成功"}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# 修改密码 changepwd 路由
@app.route("/changepwd", methods=["POST"])
def changepwd():
    try:
        data = request.get_json()
        old_password = data.get("oldPassword")
        new_password = data.get("newPassword")

        # 调用后端逻辑处理密码修改
        result = change_pwd_api(old_password, new_password)
        return jsonify({"success": True, "message": "密码修改成功"}), 200
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500





# 初始化疾病预测服务
model_weights_path = './snapshot/kappa=0.6.pth'  # 根据实际路径修改
predictor = DiseasePredictor(model_weights_path=model_weights_path)

#单人双眼检测请求
@app.route('/disease_prediction', methods=["POST","GET"])
def disease_prediction():
    if 'left_image' not in request.files:
        return jsonify({'error': 'No left eye image provided'}), 400

    left_file = request.files['left_image']
    right_file = request.files.get('right_image')  # 可选

    left_filename = secure_filename(left_file.filename)
    temp_left_image_path = './temp_left_image.jpg'
    left_file.save(temp_left_image_path)

    if right_file:
        right_filename = secure_filename(right_file.filename)
        temp_right_image_path = './temp_right_image.jpg'
        right_file.save(temp_right_image_path)
    else:
        temp_right_image_path = temp_left_image_path  # 如果没有右眼图像，使用左眼图像代替

    result_df = predictor.predict(temp_left_image_path, temp_right_image_path)

    os.remove(temp_left_image_path)
    if right_file:
        os.remove(temp_right_image_path)

    patient_id = request.form.get('patientId', 'default_patient_id')

    for index, row in result_df.iterrows():
        new_prediction = Prediction(
            patient_id=patient_id,
            left_eye_image=left_filename,
            right_eye_image=right_filename if right_file else None,
            disease_type=row['疾病类型'],
            probability=float(row['概率'])
        )
        db.session.add(new_prediction)

    db.session.commit()


    # 过滤出诊断结果为 1 且疾病类型不是 "正常" 的行
    filtered_result_df = result_df[(result_df['诊断结果'] == 1) & (result_df['疾病类型'] != "正常")]

    # 将过滤后的结果转换为 JSON 格式
    result_json = filtered_result_df.to_dict(orient='records')


    # 返回 JSON 响应
    return jsonify({
        "message": "预测成功",
        "predictions": result_json
    })


#批量检测请求
@app.route('/batch_disease_prediction', methods=["GET", "POST"])
def batch_disease_prediction():
    """
    批量疾病预测接口
    """
    # 检查是否提供了批量数据
    if 'patients' not in request.form:
        return jsonify({'error': '未提供批量患者数据'}), 400

    try:
        # 解析批量数据（JSON 格式）
        patients_data = request.form.getlist('patients')  # 获取所有患者的 JSON 数据
        patients = [eval(patient) for patient in patients_data]  # 解析为字典列表

        # 验证数据格式
        for patient in patients:
            if 'patientId' not in patient or 'left_image' not in patient:
                return jsonify({'error': '患者数据格式错误'}), 400

        # 初始化批量预测结果
        batch_results = []

        for patient in patients:
            patient_id = patient['patientId']
            left_file = patient['left_image']
            right_file = patient.get('right_image')  # 右眼图像是可选的

            # 确保文件安全
            left_filename = secure_filename(left_file.filename)
            temp_left_image_path = os.path.join('./temp_images', left_filename)
            left_file.save(temp_left_image_path)

            if right_file:
                right_filename = secure_filename(right_file.filename)
                temp_right_image_path = os.path.join('./temp_images', right_filename)
                right_file.save(temp_right_image_path)
            else:
                temp_right_image_path = temp_left_image_path  # 如果没有右眼图像，使用左眼图像代替

            # 调用预测器进行预测
            result_df = predictor.predict(temp_left_image_path, temp_right_image_path)

            # 清理临时文件
            os.remove(temp_left_image_path)
            if right_file:
                os.remove(temp_right_image_path)

            # 过滤出诊断结果为 1 且疾病类型不是 "正常" 的行
            filtered_result_df = result_df[(result_df['诊断结果'] == 1) & (result_df['疾病类型'] != "正常")]

            # 将预测结果保存到数据库
            try:
                predictions = [
                    Prediction(
                        patient_id=patient_id,
                        left_eye_image=left_filename,
                        right_eye_image=right_filename if right_file else None,
                        disease_type=row['疾病类型'],
                        probability=float(row['概率'])
                    )
                    for _, row in result_df.iterrows()
                ]
                db.session.bulk_save_objects(predictions)
                db.session.commit()
            except SQLAlchemyError as e:
                db.session.rollback()
                return jsonify({'error': f'数据库操作失败: {str(e)}'}), 500

            # 构造当前患者的返回结果
            patient_result = {
                "patient_id": patient_id,
                "left_eye_image": left_filename,
                "right_eye_image": right_filename if right_file else None,
                "predictions": filtered_result_df.to_dict(orient='records')
            }
            batch_results.append(patient_result)

        # 返回批量预测结果
        return jsonify({
            "message": "批量预测成功",
            "results": batch_results
        })

    except Exception as e:
        return jsonify({'error': f'批量预测失败: {str(e)}'}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0",port="3000")