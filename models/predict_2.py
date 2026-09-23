import os
os.environ['KMP_DUPLICATE_LIB_OK']  = 'TRUE'  # 允许重复加载
os.environ['OMP_NUM_THREADS']  = '1'         # 限制线程数

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from model.maxvit import MaxViTTwoInput

# 设置中文显示和绘图样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

LABELS = ["正常", "糖尿病", "青光眼", "白内障", "AMD", "高血压", "近视", "其他异常"]


class DiseasePredictor:
    def __init__(self, model_weights_path=None, device="cuda", threshold=0.5):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.model = MaxViTTwoInput(len(LABELS)).to(self.device)

        # 加载预训练权重（示例代码使用随机初始化）
        if model_weights_path and os.path.exists(model_weights_path):
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, left_img_path, right_img_path=None, visualize=False, save_path=None):
        """单患者预测"""
        try:
            # 加载图像
            left_img = Image.open(left_img_path).convert('RGB')
            right_img = (Image.open(right_img_path).convert('RGB') if right_img_path
                         else left_img.transpose(Image.FLIP_LEFT_RIGHT))

            # 预处理
            left_tensor = self.transform(left_img).unsqueeze(0).to(self.device)
            right_tensor = self.transform(right_img).unsqueeze(0).to(self.device)

            # 推理
            with torch.no_grad():
                logits = self.model(left_tensor, right_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                # 生成结果
            df = pd.DataFrame({
                "疾病类型": LABELS,
                "概率": np.round(probs, 4),
                "诊断结果": (probs >= self.threshold).astype(int)
            })

            # 可视化
            if visualize:
                self._visualize_prediction(left_img, right_img, df, save_path)

            return df

        except Exception as e:
            print(f"预测失败: {str(e)}")
            return None

    def batch_predict(self, patient_list, output_dir="./reports"):
        """批量预测"""
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        for idx, (left_path, right_path) in enumerate(patient_list, 1):
            pid = f"patient_{idx:03d}"
            try:
                df = self.predict(left_path, right_path, visualize=True,
                                  save_path=os.path.join(output_dir, f"{pid}_report.png"))
                results[pid] = df
                print(f"{pid} 预测完成")
            except:
                results[pid] = None
                print(f"{pid} 预测失败")

        return results

    def _visualize_prediction(self, left_img, right_img, df, save_path):
        """可视化结果"""
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1])
        # 概率分布
        ax2 = plt.subplot(gs[1, :])
        probs = df['概率'].values
        colors = ['#1f77b4' if p < self.threshold else '#ff7f0e' for p in probs]
        bars = ax2.barh(LABELS, probs, color=colors, edgecolor='black')
        ax2.axvline(self.threshold, color='r', linestyle='--')
        ax2.set_title(f' 疾病概率分布 (阈值={self.threshold})', fontsize=14)

        # 保存或显示
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    # 初始化预测器
    predictor = DiseasePredictor("snapshot/4.8 kappa=0.61.pth")

    # 准备测试数据（需准备实际图像文件）
    test_patients = [
        ("val_predict/005AMD/788_left.jpg","val_predict/005AMD/788_right.jpg" ),  # 真实双眼
        ("val_predict/003青光眼/4827_left.jpg", "val_predict/003青光眼/4826_right.jpg"),
    ]

    # 执行批量预测
    results = predictor.batch_predict(test_patients)

    # 打印结果示例
    for pid, df in results.items():
        if df is not None:
            print(f"\n{pid} 检测到异常:")
            print(df[df['诊断结果'] == 1][['疾病类型', '概率']])