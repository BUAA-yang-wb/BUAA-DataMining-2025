import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from tqdm import tqdm
import csv
import time
import math

import config
from dataset import AnomalyDataset, get_transforms
from model import Autoencoder, ResNetAutoencoder

# SSIM Loss 实现（如果配置启用）
def ssim(img1, img2, window_size=11, size_average=True):
    """
    计算结构相似性指数 (SSIM)
    img1, img2: [B, C, H, W] 范围 [0, 1]
    """
    channel = img1.size(1)
    
    # 创建高斯窗口
    def gaussian(window_size, sigma=1.5):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    _1D_window = gaussian(window_size).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device)
    
    mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = torch.nn.functional.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def train_and_evaluate(category):
    print(f"\n{'='*20} Processing Category: {category} {'='*20}")
    start_time = time.time()
    
    # 0. 准备结果保存目录
    result_dir = os.path.join(config.BASE_DIR, 'results', category)
    os.makedirs(result_dir, exist_ok=True)

    # 1. 准备数据
    transform = get_transforms()
    train_dataset = AnomalyDataset(config.DATA_ROOT, category, mode='train', transform=transform)
    test_dataset = AnomalyDataset(config.DATA_ROOT, category, mode='test', transform=transform, label_file=config.LABEL_FILE)
    
    if len(train_dataset) == 0:
        print(f"Skipping {category} due to empty training set.")
        return 0.5

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 2. 初始化模型
    if config.USE_RESNET:
        print(f"Using ResNet18-based Autoencoder (Pretrained on ImageNet)")
        model = ResNetAutoencoder(pretrained=True, freeze_encoder_layers=True).to(config.DEVICE)
        # ResNet 模型使用更小的学习率
        lr = 1e-4 if category == 'zipper' else config.LEARNING_RATE
    else:
        print(f"Using standard Autoencoder")
        model = Autoencoder().to(config.DEVICE)
        lr = config.LEARNING_RATE
    
    # 损失函数
    criterion_mse = nn.MSELoss()
    use_ssim = config.USE_SSIM_LOSS
    if use_ssim:
        print(f"Using combined loss: MSE ({1-config.SSIM_WEIGHT:.1f}) + SSIM ({config.SSIM_WEIGHT:.1f})")
    else:
        print(f"Using MSE loss only")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 添加学习率调度器，逐步降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 3. 训练循环
    print("Training...")
    loss_history = []
    model.train()
    
    # 根据类别设置不同的训练轮数
    # 对于拉链使用 ResNet 时，使用更多轮次以充分微调
    if category == 'zipper' and config.USE_RESNET:
        epochs = 150
        print(f"  → Extended training for zipper with ResNet: {epochs} epochs")
    elif category == 'zipper':
        epochs = 100
    else:
        epochs = 100
    
    # 使用 tqdm 显示训练进度
    progress_bar = tqdm(range(epochs), desc=f"Training {category}", colour='cyan')
    
    for epoch in progress_bar:
        total_loss = 0
        for images, _, _ in train_loader: # dataset now returns (image, label, path)
            images = images.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # 计算损失
            loss_mse = criterion_mse(outputs, images)
            if use_ssim:
                loss_ssim = 1 - ssim(outputs, images)
                loss = (1 - config.SSIM_WEIGHT) * loss_mse + config.SSIM_WEIGHT * loss_ssim
            else:
                loss = loss_mse
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # 学习率调度
        scheduler.step(avg_loss)
        
        # 更新进度条后缀显示当前 Loss 和学习率
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{current_lr:.6f}'})

    # 保存并绘制 Loss 曲线
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history)
    plt.title(f'{category} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(os.path.join(result_dir, 'loss_curve.png'))
    print(f"Loss curve saved to {os.path.join(result_dir, 'loss_curve.png')}")
    plt.close()

    # 保存模型权重
    torch.save(model.state_dict(), os.path.join(result_dir, 'autoencoder.pth'))

    # 4. 评估
    print("Evaluating...")
    model.eval()
    
    # 收集所有结果用于 CSV 和 直方图
    all_results = [] # list of dicts
    
    # 4.1 评估训练集 (作为 Baseline)
    # 需要一个新的 DataLoader，不打乱顺序
    train_eval_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    train_scores = []
    
    with torch.no_grad():
        for image, label, path in tqdm(train_eval_loader, desc="Evaluating Train Set", colour='green'):
            image = image.to(config.DEVICE)
            output = model(image)
            # 评估时只使用 MSE 作为异常分数
            loss = criterion_mse(output, image).item()
            
            train_scores.append(loss)
            all_results.append({
                'filename': os.path.basename(path[0]),
                'dataset': 'train',
                'true_label': 'good', # 训练集全是 good
                'anomaly_score': loss
            })

    # 4.2 评估测试集
    test_scores = []
    test_labels = []
    # 收集所有样本用于随机抽取可视化
    all_good_samples = []
    all_bad_samples = []
    
    with torch.no_grad():
        for image, label, path in tqdm(test_loader, desc="Evaluating Test Set", colour='magenta'):
            image = image.to(config.DEVICE)
            output = model(image)
            # 评估时只使用 MSE 作为异常分数
            loss = criterion_mse(output, image).item()
            lbl = label.item()
            
            test_scores.append(loss)
            test_labels.append(lbl)
            
            all_results.append({
                'filename': os.path.basename(path[0]),
                'dataset': 'test',
                'true_label': 'good' if lbl == 0 else 'bad',
                'anomaly_score': loss
            })
            
            # 收集所有样本用于后续随机选择
            if lbl == 0:
                all_good_samples.append((image.clone(), output.clone(), loss))
            else:
                all_bad_samples.append((image.clone(), output.clone(), loss))

    # 5. 保存详细结果到 CSV
    csv_path = os.path.join(result_dir, 'all_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'dataset', 'true_label', 'anomaly_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for res in all_results:
            writer.writerow(res)
    print(f"Detailed results saved to {csv_path}")

    # 6. 绘制分数分布直方图
    plt.figure(figsize=(10, 6))
    
    # 分离不同组的分数
    train_scores = np.array(train_scores)
    test_good_scores = [r['anomaly_score'] for r in all_results if r['dataset'] == 'test' and r['true_label'] == 'good']
    test_bad_scores = [r['anomaly_score'] for r in all_results if r['dataset'] == 'test' and r['true_label'] == 'bad']
    
    plt.hist(train_scores, bins=30, alpha=0.5, label='Train (Good)', color='green', density=True)
    plt.hist(test_good_scores, bins=30, alpha=0.5, label='Test (Good)', color='blue', density=True)
    plt.hist(test_bad_scores, bins=30, alpha=0.5, label='Test (Bad)', color='red', density=True)
    
    plt.title(f'{category} Anomaly Score Distribution')
    plt.xlabel('Anomaly Score (MSE)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'score_distribution.png'))
    print(f"Score distribution saved to {os.path.join(result_dir, 'score_distribution.png')}")
    plt.close()

    # 7. 可视化结果保存 (选择最后一个)
    # 选择最后一个正常样本和最后一个异常样本进行可视化
    vis_samples = {}
    if len(all_good_samples) > 0 and len(all_bad_samples) > 0:
        vis_samples['good'] = all_good_samples[-1]  # 最后一个正常样本
        vis_samples['bad'] = all_bad_samples[-1]    # 最后一个异常样本
    
    if 'good' in vis_samples and 'bad' in vis_samples:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        for i, (sample_type, data) in enumerate(vis_samples.items()):
            img_tensor, recon_tensor, score = data
            img = img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
            recon = recon_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
            residual = np.abs(img - recon)
            residual_gray = np.mean(residual, axis=2)
            
            axes[i, 0].imshow(np.clip(img, 0, 1))
            axes[i, 0].set_title(f"{sample_type.capitalize()} Input")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(np.clip(recon, 0, 1))
            axes[i, 1].set_title(f"Reconstruction (Score: {score:.4f})")
            axes[i, 1].axis('off')
            
            im = axes[i, 2].imshow(residual_gray, cmap='jet', vmin=0, vmax=1)
            axes[i, 2].set_title("Residual Error Map")
            axes[i, 2].axis('off')
            plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'visualization_result.png'))
        plt.close()

    # 8. 计算 AUROC
    try:
        auroc = roc_auc_score(test_labels, test_scores)
        print(f"Category: {category}, AUROC: {auroc:.4f}")
        
        fpr, tpr, _ = roc_curve(test_labels, test_scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUROC = {auroc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{category} ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(result_dir, 'roc_curve.png'))
        plt.close()
        
    except Exception as e:
        print(f"Error calculating AUROC: {e}")
        auroc = 0.5
    
    elapsed_time = time.time() - start_time
    print(f"\n{category} completed in {elapsed_time//60:.0f}m {elapsed_time%60:.0f}s")
    
    return auroc, elapsed_time

if __name__ == "__main__":
    total_start_time = time.time()
    results = {}
    times = {}
    
    for cat in config.CATEGORIES:
        auroc, elapsed = train_and_evaluate(cat)
        results[cat] = auroc
        times[cat] = elapsed
    
    total_elapsed = time.time() - total_start_time
    
    print("\n" + "="*60)
    print("Final Results (AUROC):")
    for cat, score in results.items():
        print(f"{cat}: {score:.4f} (Time: {times[cat]//60:.0f}m {times[cat]%60:.0f}s)")
    print(f"\nTotal Training Time: {total_elapsed//60:.0f}m {total_elapsed%60:.0f}s")
    print("="*60)
