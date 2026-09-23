# 基于眼底医学影像的眼科疾病智能诊断系统
> 团队：A07 胜利在望 | 团队编号：T2406439 | 合作方：诚迈科技（南京）股份有限公司
> English Name：*Ophthalmic Disease Intelligent Diagnosis System Based on Fundus Medical Images*

[![GitHub stars](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.3.1-orange)](https://pytorch.org/)
[![vue3](https://img.shields.io/badge/Vue-3‑TypeScript‑green.svg)](https://vuejs.org/)
[![flask](https://img.shields.io/badge/Flask‑backend‑red.svg)](https://flask.palletsprojects.com/)
[![mysql](https://img.shields.io/badge/MySQL‑8.0‑blue)](https://www.mysql.com/)

## 📖 项目简介
本系统是一套面向医疗机构的眼科疾病AI智能诊断平台，基于**MaxViT深度学习模型**对眼底医学影像做智能分析，支持单/多人双眼眼底图片识别、疾病概率预测、诊断报告PDF导出、历史记录管理、眼科智能问答助手、医疗数据可视化统计。
目标用户包含基层医院、专科眼科医院、体检中心、疾控中心、高校科研人员，辅助医生快速完成眼科疾病初筛，提升眼底影像诊断效率。

> 核心能力：上传左右眼眼底影像 → AI多标签疾病预测 → 生成PDF诊断报告 → 历史记录与数据分析；本地部署大模型实现眼科智能问答。

## ✨ 系统主要功能
1. **用户账号模块**：注册、登录、JWT身份鉴权、个人信息修改、密码重置、头像管理。
2. **眼底影像诊断**：支持单患者双眼识别、批量多人识别，图片格式支持`JPG/PNG`。
3. **AI疾病预测**：MaxViT双输入网络，同时接收左右眼眼底图，输出8类眼科疾病预测概率。
4. **报告生成导出**：基于Jinja2模板自动生成PDF诊断报告，包含患者信息、影像、疾病结果与诊疗建议。
5. **眼底图片库**：影像归档存储，检索查看历史眼底样本，支持图片增删。
6. **数据可视化分析**：疾病年龄、性别、地域、就诊时间多维度图表统计展示。
7. **眼科智能助手**：本地部署`DeepSeek‑R1‑Distill‑Qwen‑7B‑GGUF`大模型，离线实现眼科知识问答。
8. **医疗知识库**：无需登录即可查阅眼科药品、疾病、文献资料。

## 🧱 系统整体架构
> 前后端分离架构：前端Vue3 + 后端Flask + MySQL数据库 + PyTorch AI推理服务

- **应用层（Web前端）**
Vue3 + TypeScript + Pinia + Axios，包含用户管理、影像上传、结果展示、报告生成、可视化、AI助手页面。
- **服务层（Flask后端）**
用户认证服务、影像处理服务、AI预测推理服务、PDF报告生成服务；JWT无状态鉴权；bcrypt密码加密；Celery异步日志任务。
- **AI模型层**
PyTorch框架搭建MaxViT主干网络；双输入接收左右眼眼底图像；完成模型训练、评估、导出推理。
- **数据存储层（MySQL）**
用户信息表、影像信息表、诊断结果表；实现事务、索引、三级备份恢复策略。

> 完整架构图放置于 `docs/images/architecture.png`

## 🛠️ 技术栈
### 前端
- 框架：`Vue3 + TypeScript`
- 状态管理：`Pinia`
- HTTP请求：`Axios`

### 后端
- Web框架：`Flask`
- ORM：`SQLAlchemy`
- 身份认证：`JWT`
- 密码加密：`bcrypt`
- PDF模板：`Jinja2`
- 异步任务：`Celery`
- 文件处理：`Werkzeug`

### AI模型
- 深度学习框架：`PyTorch 2.3.1 + CUDA12.1`
- 主干网络：`MaxViT（Max Vision Transformer）`
- 损失函数：`BCEWithLogitsLoss`（多标签分类）
- 优化器：`Adam`
- 数据处理：`OpenCV / PIL`
- 大模型：`DeepSeek‑R1‑Distill‑Qwen‑7B‑GGUF`

### 数据库 & 部署
- 数据库：`MySQL8.0`
- Web服务器：`Nginx + Gunicorn`
- 安全：`TLS1.3 HTTPS`、AES‑256数据加密

## 📊 AI模型说明
### 数据集
- 原始赛题数据集 + 网络爬虫扩充，总数据集**10000张眼底图片**
- 划分比例：训练集70%，测试集30%
- 命名规范：左眼`left_xxx`，右眼`right_xxx`

### 数据预处理流程
1. OpenCV读取图像，裁剪有效眼球ROI区域，去除无效黑色背景
2. 训练集数据增强：随机旋转±15°、水平翻转、颜色抖动（亮度/对比度/饱和度）
3. 图像缩放至`224 × 224`
4. 转为PyTorch Tensor并归一化

### MaxViT模型配置
- 输入：双通道左右眼眼底图像 `(3,224,224)`
- 输出：8种眼科疾病分类概率（多标签分类）
- Loss：`BCEWithLogitsLoss`
- Optimizer：Adam，初始学习率 `1e‑4`
- BatchSize：16，Epoch：20
- 硬件训练环境：`RTX4090Ti / Python3.10 / CUDA12.1`

### 模型评估指标
|指标|分数|
|----|----|
|F1‑score|0.9960|
|AUC|0.9998|
|Kappa|0.7689|

> 保存最优权重文件：`model_train/snapshot/best_MaxVit.pth`，可导出TorchScript用于生产推理部署。

## 🗄️ 数据库设计
### 用户信息表 `users`
|字段|类型|说明|
|---|---|---|
|uid|int|用户ID（主键）|
|tel|varchar|手机号|
|pwd|varchar|bcrypt加密密码|
|introduce|varchar|个人简介|
|nickname|varchar|昵称|
|headpic|varchar|头像路径|
|email|varchar|邮箱|

### 诊断结果表 `predictions`
|字段|类型|说明|
|---|---|---|
|id|int|记录ID（主键）|
|patient_id|varchar|患者编号|
|left_eye_image|varchar|左眼影像文件id|
|right_eye_image|varchar|右眼影像文件id|
|disease_type|varchar|预测疾病类型|
|probability|float|预测置信概率|
|create_at|timestamp|诊断创建时间|

> 数据库备份策略：每日全量备份、每小时binlog增量备份；本地+异地云存储AES‑256加密；定义三级故障灾难恢复SLA。

## 🔌 核心API接口
> 全部接口使用`JWT Token`鉴权，传输使用HTTPS加密

|接口路径|请求方式|功能|
|---|---|---|
|`/register`|POST|用户注册|
|`/login`|POST|用户登录获取Token|
|`/getuserinfo`|GET|获取当前登录用户信息|
|`/upuserinfo`|POST|更新用户信息|
|`/changepwd`|POST|修改密码|
|`/disease_prediction`|POST|单人双眼眼底疾病预测，上传图片文件|
|`/batch_disease_prediction`|POST|批量多患者预测|
|`/chat`|POST|眼科AI助手对话接口|

## 🔐 系统安全设计
1. **身份认证**：JWT‑HS256令牌，RBAC动态角色权限控制，最小权限原则。
2. **密码安全**：bcrypt哈希存储，不保存明文密码。
3. **传输安全**：全链路TLS1.3 HTTPS加密。
4. **存储安全**：AES‑256加密敏感医疗影像与病历；密钥HSM硬件管理。
5. **上传防护**：文件后缀、文件头校验，拦截恶意脚本文件。
6. **威胁建模**：STRIDE威胁模型；防范SQL注入、XSS、重放攻击；增加请求nonce+时间戳。
7. **安全测试**：OWASP ZAP、Nessus漏洞扫描；高危漏洞24小时修复；完整操作审计日志。
8. **合规**：参考HIPAA、GDPR、国内医疗信息安全等级保护规范。

## 🚀 部署指南
### 环境依赖
```txt
python==3.10
torch==2.3.1
flask
flask‑sqlalchemy
bcrypt
pyjwt
opencv‑python
pillow
mysql‑connector‑python
jinja2
gunicorn
celery
