import torch
import torch.nn as nn
from torchvision import models



"""
# 该模型是一个基于MaxViT的双输入模型，结合了注意力融合和特征拼接的方式进行图像特征融合，最后通过分类器输出预测结果。
# 具体实现包括MaxViT网络和两种特征融合方式：CBAM和ConcatFusion。可以根据实际需求选择不同的融合方式。
"""
# 注意力融合类
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),  # 输入是通道的均值和最大值
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # 通道注意力
        channel_weights = self.channel_att(x)  # [B, C, 1, 1]
        x_channel = channel_weights * x        # [B, C, H, W]

        # 空间注意力
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_concat = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, H, W]
        spatial_weights = self.spatial_att(spatial_concat)       # [B, 1, H, W]

        # 应用空间注意力
        x_out = spatial_weights * x_channel  # [B, C, H, W]
        return x_out

# 拼接融合类
class ConcatFusion(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        # 特征归一化
        self.bn = nn.BatchNorm1d(in_features * 2)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features * 2, in_features),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, f_left, f_right):
        # 拼接特征
        concat = torch.cat((f_left, f_right), dim=1)  # [B, in_features * 2]
        # 特征归一化
        concat = self.bn(concat)
        # 分类
        out = self.fc(concat)  # [B, num_classes]
        return out
# MaxViT 双输入模型
class MaxViTTwoInput(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # 加载预训练的 MaxViT 模型
        self.maxvit = models.maxvit_t(weights=models.MaxVit_T_Weights.IMAGENET1K_V1)
        # 移除原分类头
        self.maxvit.classifier = nn.Identity()
        
        # 使用 CBAM 作为注意力模块
        self.cbam = CBAM(in_channels=512)  # MaxViT 输出特征维度为 512
        
        # 拼接融合模块
        self.concat_fusion = ConcatFusion(in_features=512, num_classes=num_classes)

    def forward(self, img1, img2):
        # 提取特征
        feat1 = self.maxvit(img1)  # [B, 512, 7, 7]
        feat2 = self.maxvit(img2)  # [B, 512, 7, 7]
        
        # 应用 CBAM 注意力
        feat1 = self.cbam(feat1)  # [B, 512, 7, 7]
        feat2 = self.cbam(feat2)  # [B, 512, 7, 7]
        
        # 全局平均池化
        feat1 = torch.mean(feat1, dim=[2, 3])  # [B, 512]
        feat2 = torch.mean(feat2, dim=[2, 3])  # [B, 512]
        
        # 拼接融合
        out = self.concat_fusion(feat1, feat2)  # [B, num_classes]
        return out