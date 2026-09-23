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
# 设置中文字体（Windows 系统）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统常用字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 使用 ggplot 样式
plt.style.use('ggplot')

LABELS = [
    "正常", "糖尿病", "青光眼", "白内障",
    "AMD", "高血压", "近视", "其他异常"
]


class DiseasePredictor:
    def __init__(self, model_weights_path, device="cuda", threshold=0.5):
        """
        初始化预测器
        :param model_weights_path: 模型权重文件路径
        :param device: 推理设备 (cuda/cpu)
        :param threshold: 疾病判定阈值 (默认0.5)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        # 初始化模型结构
        self.model = MaxViTTwoInput(num_classes=len(LABELS)).to(self.device)

        # 加载训练权重（兼容多GPU权重）
        state_dict = torch.load(model_weights_path, map_location=self.device)
        # 去除module.前缀（如果存在）
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        # 预处理流程
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_image(self, image_path):
        """加载图像并进行基本校验"""
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            raise ValueError(f"图像加载失败: {image_path}") from e

    def preprocess_image(self, image):
        """图像预处理"""
        return self.transform(image).unsqueeze(0).to(self.device)

    def _denormalize(self, tensor):
        """反标准化处理用于可视化"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor.cpu().squeeze(0) * std + mean

    def _plot_images(self, left_img, right_img, right_is_flipped=False):
        """绘制双眼图像对比"""
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        titles = ["左眼图像", "右眼图像"]
        if right_is_flipped:
            titles[1] += "\n(由左眼镜像生成)"

        for i, (img, title) in enumerate(zip([left_img, right_img], titles)):
            ax = plt.subplot(gs[i])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title, fontsize=12)

        return fig

    def _plot_probabilities(self, probs, labels, threshold):
        """绘制概率分布直方图"""
        fig = plt.figure(figsize=(12, 6))
        colors = ['#1f77b4' if p < threshold else '#ff7f0e' for p in probs]

        bars = plt.barh(range(len(labels)), probs,
                        color=colors, edgecolor='black', height=0.6)
        plt.yticks(range(len(labels)), labels, fontsize=10)
        plt.xlim(0, 1.1)
        plt.axvline(x=threshold, color='r', linestyle='--', linewidth=1)

        # 添加概率数值
        for bar, p in zip(bars, probs):
            plt.text(bar.get_width() + 0.02, bar.get_y() + 0.3,
                     f'{p:.2%}', va='center', fontsize=9)

        plt.title(f'疾病概率分布 (阈值线: {threshold})', fontsize=14)
        plt.tight_layout()
        return fig

    def _generate_incremental_path(self, original_path):
        """生成带递增编号的文件路径（类内方法）"""
        dir_name, filename = os.path.split(original_path)
        base, ext = os.path.splitext(filename)

        # 自动创建目录（若不存在）
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # 匹配已有编号文件（正则表达式优化）
        existing_files = []
        for f in os.listdir(dir_name or '.'):
            if f.startswith(base) and f.endswith(ext):
                parts = f[len(base):-len(ext)].split('_')
                if len(parts) == 1 and not parts[0]:  # 无编号的基础文件
                    existing_files.append(0)
                elif parts[-1].isdigit():
                    existing_files.append(int(parts[-1]))

                    # 确定新编号
        new_counter = max(existing_files) + 1 if existing_files else 1
        return os.path.join(dir_name, f"{base}_{new_counter:03d}{ext}")

    def visualize(self, left_img_path, df_result,
                  right_img_path=None, save_path=None):
        """
        可视化预测结果
        :param left_img_path: 左眼原始图像路径
        :param df_result: predict()返回的结果DataFrame
        :param right_img_path: 右眼原始路径（如果有）
        :param save_path: 结果保存路径（可选）
        """
        # 准备数据
        probs = df_result['概率'].values
        labels = df_result['疾病类型'].values
        threshold = self.threshold

        # 加载原始图像
        left_img = Image.open(left_img_path).convert('RGB')
        right_img = (Image.open(right_img_path).convert('RGB')
                     if right_img_path else None)

        # 创建画布
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1])

        # 绘制输入图像
        ax1 = plt.subplot(gs[0, :])
        self._plot_images(left_img,
                          right_img if right_img else left_img.transpose(Image.FLIP_LEFT_RIGHT),
                          right_is_flipped=(right_img is None))
        ax1.set_title('输入图像对比', fontsize=14)

        # 绘制概率分布
        ax2 = plt.subplot(gs[1, :])
        self._plot_probabilities(probs, labels, threshold)

        # 保存或显示
        if save_path:
            final_path = self._generate_incremental_path(save_path)  # 调用类内方法
            plt.savefig(final_path, bbox_inches='tight', dpi=150)
            print(f"可视化结果已保存至: {final_path}")
        else:
            plt.show()

        plt.close()

    def predict(self, left_img_path, right_img_path=None, visualize=False, save_path=None):
        """
        执行预测
        :param left_img_path: 左眼图像路径（必选）
        :param right_img_path: 右眼图像路径（可选）
        :param visualize: 是否生成可视化图表
        :param save_path: 可视化结果保存路径
        :return: 包含概率和诊断结果的 DataFrame
        """
        # 加载基础图像
        left_img = self._load_image(left_img_path)

        # 处理右眼图像
        if right_img_path:
            right_img = self._load_image(right_img_path)
        else:
            # 水平翻转左眼图像作为右眼输入
            right_img = left_img.transpose(Image.FLIP_LEFT_RIGHT)

        # 预处理
        left_tensor = self.preprocess_image(left_img)
        right_tensor = self.preprocess_image(right_img)

        # 推理
        with torch.no_grad():
            logits = self.model(left_tensor, right_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        # 生成结果
        df_result = pd.DataFrame({
            "疾病类型": LABELS,
            "概率": np.round(probs, 4),
            "诊断结果": (probs >= self.threshold).astype(int)
        })

        # 可视化处理
        if visualize:
            self.visualize(
                left_img_path=left_img_path,
                df_result=df_result,
                right_img_path=right_img_path,
                save_path=save_path
            )

        return df_result


# 使用示例
if __name__ == "__main__":
    # 初始化预测器
    predictor = DiseasePredictor("snapshot/4.8 kappa=0.61.pth")

    try:
        # 执行预测并生成可视化报告
        result_df = predictor.predict(
            left_img_path="./val_predict/008其他和多病/1149_left.jpg",
            right_img_path="./val_predict/008其他和多病/1149_right.jpg",
            visualize=True,
            save_path="./reports/prediction_report.png"
        )

        # 打印诊断结果
        print("\n眼底疾病预测报告：")
        print(result_df[result_df["诊断结果"] == 1][["疾病类型", "概率"]])

    except Exception as e:
        print(f"预测失败: {str(e)}")