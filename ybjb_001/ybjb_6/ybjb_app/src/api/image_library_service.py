# /app/services/image_library_service.py

import os

def image_library_algorithm():
    """
    图片库管理算法的具体实现
    :return: 图片文件名列表
    """
    upload_folder = os.getenv('UPLOAD_FOLDER')
    images = [f for f in os.listdir(upload_folder) if os.path.isfile(os.path.join(upload_folder, f))]
    return images

def get_image_list():
    """
    公共接口函数，调用具体的算法
    :return: 图片文件名列表
    """
    return image_library_algorithm()