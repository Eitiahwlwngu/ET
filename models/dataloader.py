import os
import cv2
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class TwoImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Read CSV file into a DataFrame
        self.annotations = pd.read_csv(csv_file, header=None)
        # Store the root directory path
        self.root_dir = root_dir
        # Store the transformations
        self.transform = transform

    def __len__(self):
        # Return the total number of samples
        return len(self.annotations)

    def __getitem__(self, idx):
        # Retrieve one row of data
        row_data = self.annotations.iloc[idx]

        # Retrieve the image file names from the row (first two columns)
        img_name1 = row_data[0]
        img_name2 = row_data[1]

        # Retrieve the labels from columns 3 to 10
        labels = row_data[2:10].values.astype(float)

        # Build the complete path for the two images
        img1_path = os.path.join(self.root_dir, img_name1)
        img2_path = os.path.join(self.root_dir, img_name2)
        #print(img1_path)
        #print(img2_path)

        # Open the images and convert them to RGB
        image1 = Image.open(img1_path).convert('RGB')
        image2 = Image.open(img2_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Convert labels into a PyTorch tensor
        labels = torch.tensor(labels, dtype=torch.float32)
        #print(labels)

        # Return the two images and the label tensor
        return image1, image2, labels

if __name__ == "__main__":
    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),         # Convert PIL images to tensors
    ])
    
    # Create an instance of the custom dataset
    dataset = TwoImageDataset(
        csv_file="./dataset/ODIR/train_anno.csv",
        root_dir="./dataset/ODIR/train/Images/",
        transform=transform
    )

    # Create a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,     # Number of samples in each batch
        shuffle=False,     # Shuffle the data
        num_workers=0     # Number of subprocesses for data loading
    )

    # Iterate through the DataLoader
    for batch_idx, (img1, img2, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(" - img1 shape:", img1.shape)
        print(" - img2 shape:", img2.shape)
        print(" - labels shape:", labels.shape)
        # Here you can implement your training/validation logic
        break  # Just for demonstration, we break after the first batch



def mixup_data_dual(img1, img2, labels, alpha=1.0, device=None):
    """
    对双输入样本进行 Mixup 增强。
    
    Args:
        img1: Tensor, shape [batch_size, C, H, W]，第一张图像
        img2: Tensor, shape [batch_size, C, H, W]，第二张图像
        labels: Tensor, shape [batch_size, ...]，标签（多标签时一般为 float，或 one-hot 编码）
        alpha: Mixup 超参数（若 alpha<=0 则不做 Mixup）
        device: 如果需要指定设备（一般 img1、img2 已在同一设备上）
        
    Returns:
        mixed_img1, mixed_img2, mixed_labels, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = img1.size(0)
    if device is None:
        device = img1.device
    index = torch.randperm(batch_size).to(device)

    mixed_img1 = lam * img1 + (1 - lam) * img1[index, :]
    mixed_img2 = lam * img2 + (1 - lam) * img2[index, :]
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    return mixed_img1, mixed_img2, mixed_labels, lam

def rand_bbox(size, lam):
    """
    生成 CutMix 随机裁剪区域的坐标。
    
    Args:
        size: 输入张量的形状，形如 [batch_size, C, H, W]
        lam: mix 参数
    
    Returns:
        bbx1, bby1, bbx2, bby2: 裁剪区域的左上和右下坐标
    """
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)
    
    # 随机选择裁剪中心
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data_dual(img1, img2, labels, alpha=1.0, device=None):
    """
    对双输入样本进行 CutMix 增强。
    
    Args:
        img1: Tensor, shape [batch_size, C, H, W]，第一张图像
        img2: Tensor, shape [batch_size, C, H, W]，第二张图像
        labels: Tensor, shape [batch_size, ...]，标签（一般为 float 类型）
        alpha: CutMix 超参数（若 alpha<=0 则不做 CutMix）
        device: 如果需要指定设备
        
    Returns:
        img1, img2, mixed_labels, lam，其中 img1 和 img2 都已经经过 CutMix 操作
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = img1.size(0)
    if device is None:
        device = img1.device
    index = torch.randperm(batch_size).to(device)
    
    # 生成裁剪区域
    bbx1, bby1, bbx2, bby2 = rand_bbox(img1.size(), lam)
    
    # 对两组图像都执行 CutMix：用同一裁剪区域和同一 permutation
    img1[:, :, bby1:bby2, bbx1:bbx2] = img1[index, :, bby1:bby2, bbx1:bbx2]
    img2[:, :, bby1:bby2, bbx1:bbx2] = img2[index, :, bby1:bby2, bbx1:bbx2]
    
    # 根据替换区域重新计算 lam
    lam_adjusted = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img1.size(-1) * img1.size(-2)))
    
    mixed_labels = lam_adjusted * labels + (1 - lam_adjusted) * labels[index]
    return img1, img2, mixed_labels, lam_adjusted
