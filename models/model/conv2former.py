import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

"""
# 整体说明：
# 该脚本实现了一个基于卷积和变压器混合架构的视觉模型 Conv2Former。
该模型结合了卷积层和变压器（Transformer）结构，利用卷积进行空间信息处理，再利用变压器进行全局信息建模。
# 该模型适用于图像分类任务，具有可扩展性，并通过多个子模块（如MLP，空间注意力等）构建不同的功能模块。
多个模型版本也被注册在 `@register_model` 装饰器中，以供不同的预训练权重使用。
"""

class MLP(nn.Module):
    """
    多层感知机（MLP）模块，包含两个卷积层，中间使用激活函数和位置卷积。
    该模块的作用是通过非线性变换增强特征的表达能力。
    """


    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")  # 层归一化
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)  # 第一个卷积层，扩展特征维度
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)  # 位置卷积, 用于位置信息的融合
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)  # 第二个卷积层，恢复特征维度
        self.act = nn.GELU()  # 激活函数

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)  # 对输入进行层归一化
        x = self.fc1(x)  # 第一个卷积层
        x = self.act(x)   # 激活
        x = x + self.act(self.pos(x))  # 加上位置卷积的输出
        x = self.fc2(x)  # 第二个卷积层

        return x

class SpatialAttention(nn.Module):
    """
    空间注意力模块，通过卷积和加权操作增强特征图的空间信息。
    """
    def __init__(self, dim, kernel_size, expand_ratio=2):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")  # 层归一化
        self.att = nn.Sequential(
                nn.Conv2d(dim, dim, 1),  # 卷积层1，改变通道数
                nn.GELU(),  # 激活函数
                nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)  # 空间卷积
        )
        self.v = nn.Conv2d(dim, dim, 1)  # 卷积层v，用于加权
        self.proj = nn.Conv2d(dim, dim, 1)  # 最终投影

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)           # 层归一化
        x = self.att(x) * self.v(x)   # 通过空间卷积和加权操作
        x = self.proj(x)  # 投影
        return x

class Block(nn.Module):
    """
    基于空间注意力和MLP的残差块。通过drop path进行正则化。
    """
    def __init__(self, index, dim, kernel_size, num_head, window_size=14, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.attn = SpatialAttention(dim, kernel_size)    # 空间注意力模块
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # DropPath正则化
        self.mlp = MLP(dim, mlp_ratio)  # MLP模块
        layer_scale_init_value = 1e-6   # 初始层缩放值
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)  # 第一层缩放
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)  # 第二层缩放

    def forward(self, x):
        # 残差连接，使用DropPath进行正则化
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

class Conv2Former(nn.Module):
    """
    Conv2Former模型，结合了卷积和变压器（Transformer）结构进行图像分类任务。
    包括多个残差块、Downsample层以及自定义的MLP和空间注意力模块。
    """
    def __init__(self, kernel_size,
                 img_size=224,
                 in_chans=3,
                 num_classes=1000,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 window_sizes=[14, 14, 14, 7],
                 mlp_ratios=[4, 4, 4, 4],
                 num_heads=[2, 4, 10, 16],
                 layer_scale_init_value=1e-6,
                 head_init_scale=1.,
                 drop_path_rate=0.,
                 drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # 构建Downsample层
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(dims[0] // 2),
            nn.Conv2d(dims[0] // 2, dims[0] // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(dims[0] // 2),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=2, stride=2, bias=False),
        )
        self.downsample_layers.append(stem)
        for i in range(len(dims)-1):
            stride = 2
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    # nn.Conv2d(dims[i], dims[i+1], 1),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=stride, stride=stride),
            )
            self.downsample_layers.append(downsample_layer)

        # 构建各个阶段的残差块
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[Block(index=cur+j, dim=dims[i], kernel_size=kernel_size, drop_path=dp_rates[cur + j], num_head=num_heads[i], window_size=window_sizes[i], mlp_ratio=mlp_ratios[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # 最终的head模块
        self.head = nn.Sequential(
                nn.Conv2d(dims[-1], 1280, 1),
                nn.GELU(),
                LayerNorm(1280, eps=1e-6, data_format="channels_first")
        )
        # 最终分类器
        self.pred = nn.Linear(1280, num_classes)
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02) # 使用trunc_normal_初始化权重
            # nn.init.constant_(m.bias, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm): # 初始化LayerNorm，bias为0
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # 输入图像通过Downsample层和多个阶段的残差块进行处理
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.head(x)  # 通过head模块进行处理
        return x.mean([-2, -1]) # 全局平均池化, (N, C, H, W) -> (N, C)

    def forward(self, x):
        # 模型的前向传播：提取特征并进行分类
        x = self.forward_features(x)
        x = self.pred(x)  # 最终分类
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    """
    支持两种数据格式的LayerNorm。可以选择 "channels_last" 或 "channels_first"。
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        # 根据数据格式选择合适的LayerNorm处理方式
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)  # 计算均值
            s = (x - u).pow(2).mean(1, keepdim=True)  # 计算方差
            x = (x - u) / torch.sqrt(s + self.eps)  # 归一化
            x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 乘以权重和偏置
            return x

# 注册不同配置的模型版本
@register_model
def conv2former_n(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=7, dims=[64, 128, 256, 512], mlp_ratios=[4, 4, 4, 4], depths=[2, 2, 8, 2], **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def conv2former_t(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=11, dims=[72, 144, 288, 576], mlp_ratios=[4, 4, 4, 4], depths=[3, 3, 12, 3], **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def conv2former_s(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=11, dims=[72, 144, 288, 576], mlp_ratios=[4, 4, 4, 4], depths=[4, 4, 32, 4], **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def conv2former_b(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=11, dims=[96, 192, 384, 768], mlp_ratios=[4, 4, 4, 4], depths=[4, 4, 34, 4], **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def conv2former_b_22k(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=7, dims=[96, 192, 384, 768], mlp_ratios=[4, 4, 4, 4], depths=[4, 4, 34, 4], **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def conv2former_l(pretrained=False, **kwargs):
    model = Conv2Former(kernel_size=11, dims=[128, 256, 512, 1024], mlp_ratios=[4, 4, 4, 4], depths=[4, 4, 48, 4], **kwargs)
    model.default_cfg = _cfg()
    return model


class Conv2FormerTwoInput(nn.Module):
    """
    一个扩展版本的Conv2Former，用于处理两个输入图像，并将其特征进行连接（concat）后进行分类。
    """
    def __init__(self, num_classes=8):
        super(Conv2FormerTwoInput, self).__init__()
        self.net = conv2former_n(num_classes=1000)   # 使用预训练的Conv2Former模型
      # self.net.load_state_dict(torch.load('./snapshot/best_swin.pth'))

        in_features = 2560//2  # 2560是Conv2Former的输出特征维度，除以2是为了得到两个输入的特征维度，即每个输入的特征维度为1280

        # Our custom classifier for 2 images => concat => 2 * in_features
        # 自定义分类器，将两个输入的特征拼接后进行分类
        self.classifier = nn.Linear(in_features * 2, num_classes)

    def forward(self, img1, img2):
        feat1 = self.net.forward_features(img1)  # shape: [B, 1024]， # 获取第一个图像的特征
        feat2 = self.net.forward_features(img2)  # shape: [B, 1024]， # 获取第二个图像的特征

        combined = torch.cat((feat1, feat2), dim=1)  # [B, 2048]，# 拼接特征
        out = self.classifier(combined)              # [B, num_classes]， # 通过分类器输出类别概率
        return out
