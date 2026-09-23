#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from model.maxvit import MaxViTTwoInput as ODIR_Model
#from efficientnet import EfficientNetTwoInput as ODIRModel
#from model.vgg import VGGTwoInput as ODIR_Model
#from model.swin import SwinTransformerTwoInput as ODIR_Model
from dataloader import TwoImageDataset
from utils import ODIR_Metrics,compute_label_based_metrics, ODIR_Metrics2

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    N = len(dataloader.dataset)  # 验证集总样本数
    L = 8  # 假设你的多标签数是8，可从数据集/模型推断

    all_preds = torch.zeros((N, L), dtype=torch.float32)   # 存放预测标签(0/1)
    all_labels = torch.zeros((N, L), dtype=torch.long)  # 存放真实标签(0/1)

    start_idx = 0
    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            outputs = model(img1, img2)  # shape: [batch_size, L]
            loss = criterion(outputs, labels)
            running_loss += loss.item() * img1.size(0)

            probs = torch.sigmoid(outputs)
            batch_size_now = probs.size(0)  
            all_preds[start_idx : start_idx + batch_size_now, :] = probs.cpu()
            all_labels[start_idx : start_idx + batch_size_now, :] = labels.cpu().long()
            start_idx += batch_size_now

    threshold = 0.5
    #acc, prec, rec = compute_label_based_metrics(all_labels, all_preds, threshold)
    #kappa, f1, auc = ODIR_Metrics(all_labels, all_preds, threshold)
    kappa, f1, auc, acc, prec, rec = ODIR_Metrics2(all_labels, all_preds, threshold)
    return kappa, f1, auc, acc, prec, rec

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    batch_size = 8

    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # 固定大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    valid_dataset = TwoImageDataset(csv_file="./dataset/ODIR/offsite_anno.csv", root_dir="./dataset/ODIR/offsite/Images_Crop/", transform=valid_transforms)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print('Valid images: %s' % len(valid_dataloader.dataset))

    # Initialize model, loss function, and optimizer
    model = ODIR_Model(num_classes=8).to(device)
    criterion = nn.BCEWithLogitsLoss()

    model.load_state_dict(torch.load('./snapshot/kappa=0.6.pth', map_location=device))

    kappa, f1, auc, acc, prec, rec  = validate_one_epoch(model, dataloader=valid_dataloader, criterion=criterion, device=device)
    print(f"Kappa: {kappa:.4f} | F1-score: {f1:.4f} | AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

if __name__ == "__main__":
    main()
