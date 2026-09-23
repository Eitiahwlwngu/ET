import torch
import torch.nn as nn
import torch.nn.functional as F

"""
本脚本实现了 Focal Loss，一种用于二分类或多标签分类的损失函数。
该损失函数在难分类样本上给予更高的权重，减少对易分类样本的关注，帮助模型聚焦于难分类样本，防止在样本不均衡的情况下训练效果不好
"""

class FocalLoss(nn.Module):
    """
    用于多标签或二分类的Focal Loss。

    参数：
    -------
    alpha (float or list): 稀有类的权重因子，可以是单个浮动值或者每个类别的权重列表/张量。
    gamma (float): 聚焦参数，用于减少对分类准确样本的损失。
    reduction (str): 处理损失的方式。'mean'：所有元素的平均值；'sum'：所有元素的和；'none'：每个元素返回损失值。

    使用示例：
    ----------
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        logits = model(inputs)  # [batch_size, num_labels]
        loss = criterion(logits, labels)  # [batch_size, num_labels] -> scalar
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        初始化Focal Loss。

        参数：
        alpha (float or list): 影响类别稀有性的权重因子。
        gamma (float): 聚焦参数，用于减少分类正确的样本损失。
        reduction (str): 损失值的处理方式（'mean'、'sum'、'none'）。
        """
        super(FocalLoss, self).__init__()
        # 如果alpha是浮动值，则直接赋值，如果alpha是每个类别的权重列表/张量，转换为tensor
        if isinstance(alpha, float):
            self.alpha = alpha
        elif isinstance(alpha, (list, torch.Tensor)):
            # If alpha is per-label weighting, convert to tensor
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            raise TypeError("alpha must be float or list/torch.Tensor.")

        self.gamma = gamma   # 聚焦参数gamma
        self.reduction = reduction  # 损失值的处理方式

    def forward(self, logits, targets):
        """
        前向传播计算Focal Loss。

        参数：
        logits: 模型输出的原始预测值（[batch_size, num_labels]），没有经过sigmoid。
        targets: 目标标签（[batch_size, num_labels]），标签为0或1。

        返回：
        计算的损失值，取决于reduction的设置。
        """
        # logits => sigmoid => p, 转换为预测概率
        p = torch.sigmoid(logits)
        # p, targets shape => [N, L] or [N]

        # cross entropy part: -[ y * log(p) + (1-y)*log(1-p) ]
        # 使用二元交叉熵损失计算
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none'
        )  # shape: [N, L]

        # 将目标转换为float类型
        targets = targets.float()

        # p_t: p if y=1 else (1-p)
        # same shape [N, L]
        # 计算p_t: 如果y=1，则p_t=p，否则p_t=1-p
        p_t = p * targets + (1 - p) * (1 - targets)

        # focal因子：alpha * (1 - p_t)^gamma
        # 如果alpha是每个类别的，扩展它的维度
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.dim() == 0:  
                # 如果是单一alpha值
                alpha_t = self.alpha
            else:
                # 如果是每个类别的alpha，扩展为[1, L]
                alpha_t = self.alpha.unsqueeze(0)  # shape => [1, L]
        else:
            alpha_t = self.alpha

        # 计算focal因子
        focal_factor = alpha_t * ((1 - p_t) ** self.gamma)

        # 最终的focal损失
        focal_loss = focal_factor * ce_loss  # [N, L]

        # 根据reduction方式，返回最终的损失
        if self.reduction == 'mean':
            return focal_loss.mean()   # 返回所有元素的平均损失
        elif self.reduction == 'sum':
            return focal_loss.sum()   # 返回所有元素的和
        else:
            # 'none' 返回每个元素的损失
            return focal_loss
