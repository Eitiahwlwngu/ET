import torch
import torch.nn as nn
import torch.nn.functional as F
"""
# 整体说明：该脚本实现了一个结合了BCE（Binary Cross Entropy）损失和Dice损失的复合损失函数。
# 这个损失函数可以用于多标签分类或语义分割任务。通过加权的方式同时优化BCE损失和Dice损失，
# 在处理类别不平衡或目标区域较小的图像时尤其有效。Dice损失可以帮助模型更好地捕捉到目标区域的形状。
"""


def dice_loss(prob, target, eps=1e-6):
    """
    计算Dice损失（1 - Dice系数），用于衡量预测和真实标签的重叠程度。Dice系数越高，表示重叠区域越大，损失越小。

    参数：
    - prob: (N, C, H, W) 或 (N, C) - 预测的概率，通常是经过sigmoid或softmax的结果
    - target: (N, C, H, W) 或 (N, C) - 真实标签，值为0或1
    - eps: 防止除以零的小常数

    返回：
    - 计算的Dice损失，1 - Dice系数
    """
    # prob 和 target 都是 float 类型
    # 多通道情形下，对每个通道分别计算Dice，再求平均
    # 如果只是二维(N, C)，则视作batch= N, channel=C
    
    # 在计算前，先把数据展平 (N, C, H, W) -> (N*C, H*W)，简化求和
    prob_flat = prob.view(prob.size(0), prob.size(1), -1)   # [N, C, -1]，展平为 [N, C, H*W]
    prob_flat = prob_flat.view(-1, prob_flat.size(-1))     # # [N*C, H*W]，每个通道变为一个向量

    target_flat = target.view(target.size(0), target.size(1), -1)   # 同样展平目标图像
    target_flat = target_flat.view(-1, target_flat.size(-1))  # [N*C, H*W]

    # 计算交集和并集
    intersection = (prob_flat * target_flat).sum(dim=1)   # 计算交集 [N*C]
    union = prob_flat.sum(dim=1) + target_flat.sum(dim=1) # 计算并集 [N*C]
    dice = (2. * intersection + eps) / (union + eps)      # Dice系数[N*C]，加eps防止除零

    dice_mean = dice.mean()  # 计算所有通道的平均Dice系数
    return 1. - dice_mean  # 返回Dice损失，值为1-Dice系数


class BCEDiceLoss(nn.Module):
    """
    结合BCE损失和Dice损失的复合损失函数。该损失函数适用于多标签分类和语义分割任务，只要输出与目标 shapes 对应.

    参数:
    alpha: float, 在[0, 1]之间，控制BCE损失与Dice损失的加权比例。alpha越大，BCE损失占主导作用；反之，Dice损失占主导作用。
    """
    def __init__(self, alpha=0.5):
        super(BCEDiceLoss, self).__init__()  # 初始化父类
        self.alpha = alpha  # 保存alpha的值，用于加权
        self.bce = nn.BCEWithLogitsLoss()  # 初始化BCE损失函数（自动处理sigmoid）

    def forward(self, logits, targets):
        """
        计算BCEDice损失，结合BCE损失和Dice损失。
        参数：
        - logits: 模型输出，形状为[N, C, ...]（不需要经过sigmoid，因为BCEWithLogitsLoss会自动处理）
        - targets: 真实标签，形状为[N, C, ...]，值为0或1
        返回：
        - 结合了BCE和Dice的加权损失
        """

        # 1) 先计算BCE损失
        loss_bce = self.bce(logits, targets.float())  # 将目标标签转换为浮动类型，计算BCE损失

        # 2) 计算Dice损失
        #    但DiceLoss需要概率 => 先做 sigmoid
        prob = torch.sigmoid(logits)  # 对logits应用sigmoid函数，得到预测概率
        loss_dice = dice_loss(prob, targets.float())  # 计算Dice损失

        # 3) 加权求和
        loss = self.alpha * loss_bce + (1 - self.alpha) * loss_dice  # 根据alpha加权BCE损失和Dice损失
        return loss  # 返回最终的加权损失
