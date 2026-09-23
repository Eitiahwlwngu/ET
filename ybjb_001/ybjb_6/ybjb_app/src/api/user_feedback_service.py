# /app/services/user_feedback_service.py

import os
from datetime import datetime

FEEDBACK_FILE = os.path.join(os.getcwd(), 'flask_app/static/feedbacks.txt')

def user_feedback_algorithm(feedback):
    """
    用户反馈处理算法的具体实现
    :param feedback: 用户反馈的内容（字符串）
    """
    with open(FEEDBACK_FILE, 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{timestamp}: {feedback}\n")

def save_feedback(feedback):
    """
    公共接口函数，调用具体的算法
    :param feedback: 用户反馈的内容（字符串）
    """
    user_feedback_algorithm(feedback)