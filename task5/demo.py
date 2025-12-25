"""
多模态异常检测Transformer演示脚本
展示如何使用统一架构同时处理图像和数值异常检测
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import os
import argparse
import time
from typing import Dict, Any
from torch.utils.data import DataLoader
from tqdm import tqdm

from multimodal_anomaly_detector import (
    MultiModalAnomalyTransformer, Trainer, load_image_anomaly_data,
    load_thyroid_data, create_dataloaders, MultiModalDataset
)
from config import get_config, get_small_config, get_large_config
from visualization import AnomalyVisualizer
from logger import setup_experiment_logging, BatchTimer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多模态异常检测Transformer演示')

    parser.add_argument('--config', type=str, default='default',
                       choices=['small', 'default', 'large'],
                       help='模型配置大小')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数（覆盖配置文件）')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小（覆盖配置文件）')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='计算设备')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='结果保存目录')
    parser.add_argument('--quick_test', action='store_true',
                       help='快速测试模式（减少训练轮数）')
    parser.add_argument('--resume', type=str, default=None,
                       help='从checkpoint继续训练的路径')

    return parser.parse_args()


def setup_device(device_str: str):
    """设置计算设备"""
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(device)}")
        print(f"CUDA版本: {torch.version.cuda}")

    return device


def load_data(config):
    """加载数据"""
    print("加载数据...")

    # 加载图像数据
    image_train_paths, image_test_paths, image_test_labels = load_image_anomaly_data(
        config.image_data_path, config.image_category, config.train_normal_only
    )

    # 加载数值数据
    numerical_train_data, numerical_test_data, numerical_test_labels = load_thyroid_data(
        config.thyroid_data_path
    )

    print("数据集统计:")
    print(f"  图像训练样本: {len(image_train_paths)} (仅正常样本)")
    print(f"  图像测试样本: {len(image_test_paths)}")
    print(f"  数值训练样本: {len(numerical_train_data)} (仅正常样本)")
    print(f"  数值测试样本: {len(numerical_test_data)}")

    # 计算异常比例
    image_anomaly_ratio = sum(image_test_labels) / len(image_test_labels) if image_test_labels else 0
    numerical_anomaly_ratio = sum(numerical_test_labels) / len(numerical_test_labels)

    print(".2%")
    print(".2%")

    return (image_train_paths, image_test_paths, image_test_labels,
            numerical_train_data, numerical_test_data, numerical_test_labels)


def create_model_and_trainer(model_config, training_config, device):
    """创建模型和训练器"""
    print("创建模型...")

    model = MultiModalAnomalyTransformer(
        img_size=model_config.img_size,
        patch_size=model_config.patch_size,
        in_chans=model_config.in_chans,
        num_numerical_features=model_config.num_numerical_features,
        embed_dim=model_config.embed_dim,
        depth=model_config.depth,
        num_heads=model_config.num_heads,
        mlp_ratio=model_config.mlp_ratio,
        qkv_bias=model_config.qkv_bias,
        drop_rate=model_config.drop_rate,
        attn_drop_rate=model_config.attn_drop_rate
    )

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📏 模型信息:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    trainer = Trainer(model, device)
    trainer.optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=training_config.num_epochs
    )

    return model, trainer


def train_model(trainer, train_loader, num_epochs, save_dir):
    """训练模型"""
    print("🚀 开始训练...")

    train_losses = []
    val_aucs = []

    for epoch in range(num_epochs):
        # 训练一个epoch
        train_loss = trainer.train_epoch(train_loader)
        train_losses.append(train_loss)

        print("2d")

        # 每5个epoch进行一次验证
        if (epoch + 1) % 5 == 0:
            try:
                # 这里可以添加验证逻辑
                val_auc = 0.0  # 暂时设为0
                val_aucs.append(val_auc)
            except:
                val_aucs.append(0.0)

    print("训练完成！")
    return train_losses, val_aucs


def evaluate_model(trainer, test_loader, visualizer, threshold_percentile=95.0):
    """评估模型"""
    print("评估模型...")

    # 获取预测分数
    scores, labels = trainer.evaluate(test_loader)

    # 计算AUC
    auc_score = roc_auc_score(labels, scores)

    # 计算最佳阈值和预测结果
    threshold = np.percentile(scores, threshold_percentile)
    predictions = (scores > threshold).astype(int)

    # 计算分类指标
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)


    # 可视化结果
    print("生成可视化图表...")

    visualizer.plot_anomaly_scores(scores, labels)
    auc_val, optimal_threshold = visualizer.plot_roc_curve(labels, scores)
    ap_score = visualizer.plot_precision_recall_curve(labels, scores)

    # 使用最佳阈值重新计算预测结果
    optimal_predictions = (scores > optimal_threshold).astype(int)
    visualizer.plot_confusion_matrix(labels, optimal_predictions)

    # 计算最佳阈值下的指标
    opt_precision, opt_recall, opt_f1, _ = precision_recall_fscore_support(
        labels, optimal_predictions, average='binary'
    )
    opt_accuracy = accuracy_score(labels, optimal_predictions)


    return {
        'auc': auc_score,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'optimal_threshold': optimal_threshold,
        'used_threshold': threshold,
        'ap_score': ap_score,
        'opt_precision': opt_precision,
        'opt_recall': opt_recall,
        'opt_f1': opt_f1,
        'opt_accuracy': opt_accuracy,
        'scores': scores,
        'labels': labels,
        'predictions': predictions,
        'optimal_predictions': optimal_predictions
    }


def save_results(results: Dict[str, Any], config, save_dir: str):
    """保存结果"""
    os.makedirs(save_dir, exist_ok=True)

    # 保存模型
    torch.save({
        'model_state_dict': results['model'].state_dict(),
        'config': config,
        'task_results': results.get('task_results', {})
    }, os.path.join(save_dir, 'model_checkpoint.pth'))

    # 保存配置
    with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
        f.write("模型配置:\n")
        for key, value in vars(config).items():
            f.write(f"  {key}: {value}\n")

    print(f"结果已保存到: {save_dir}")


def train_task_separately(trainer, image_train_loader, numerical_train_loader, image_val_loader, numerical_val_loader, num_epochs, save_dir):
    """分别训练两种任务"""
    print("分别训练Task2和Task4...")
    print("=" * 50)

    # 设置日志记录器
    logger = setup_experiment_logging(save_dir, f"multimodal_training_{time.strftime('%Y%m%d_%H%M%S')}")
    batch_timer = BatchTimer()

    # 初始化最佳模型跟踪
    best_task2_auc = 0.0
    best_task4_auc = 0.0
    best_combined_auc = 0.0
    best_epoch = 0

    # 记录训练配置
    train_config = {
        "num_epochs": num_epochs,
        "image_train_batches": len(image_train_loader) if image_train_loader else 0,
        "numerical_train_batches": len(numerical_train_loader) if numerical_train_loader else 0,
        "image_val_batches": len(image_val_loader) if image_val_loader else 0,
        "numerical_val_batches": len(numerical_val_loader) if numerical_val_loader else 0,
        "learning_rate": trainer.optimizer.param_groups[0]['lr']
    }
    logger.log_config(train_config)

    all_train_losses = []

    # 交替训练两种模态
    for epoch in range(num_epochs):
        epoch_losses = []
        batch_timer.reset()

        logger.log_epoch_start(epoch, num_epochs)

        # 设置训练轮次平衡（图像数据量少，多训练几轮）
        image_epochs = 2  # 图像任务训练1轮（每个epoch训练一次）
        numerical_epochs = 1  # 数值任务训练1轮

        logger.log_info(f"Epoch {epoch + 1}: 图像训练轮次={image_epochs}, 数值训练轮次={numerical_epochs}")

        # 训练Task2：图像异常检测
        logger.log_task_start("Task2图像训练", f"Epoch {epoch + 1}")

        image_batch_losses = []
        if image_train_loader:
            for epoch_i in range(image_epochs):
                for batch_idx, batch in enumerate(tqdm(image_train_loader, desc=f"    图像轮次{epoch_i+1}", leave=False)):
                    trainer.optimizer.zero_grad()

                    image_data = batch.get('image').to(trainer.device) if batch.get('image') is not None else None

                    reconstruction_loss, anomaly_score, _, _ = trainer.model(image_data=image_data)
                    loss = reconstruction_loss.mean() + anomaly_score.mean()

                    loss.backward()
                    trainer.optimizer.step()

                    loss_val = loss.item()
                    epoch_losses.append(loss_val)
                    image_batch_losses.append(loss_val)

        avg_image_loss = sum(image_batch_losses) / len(image_batch_losses) if image_batch_losses else 0.0
        logger.log_task_end("Task2图像训练", results={"avg_loss": avg_image_loss, "epochs": image_epochs})

        # 训练Task4：数值异常检测
        logger.log_task_start("Task4数值训练", f"Epoch {epoch + 1}")

        numerical_batch_losses = []
        if numerical_train_loader:
            for epoch_i in range(numerical_epochs):
                for batch_idx, batch in enumerate(tqdm(numerical_train_loader, desc=f"    数值轮次{epoch_i+1}", leave=False)):
                    trainer.optimizer.zero_grad()

                    numerical_data = batch.get('numerical').to(trainer.device) if batch.get('numerical') is not None else None

                    reconstruction_loss, anomaly_score, _, _ = trainer.model(numerical_data=numerical_data)
                    loss = reconstruction_loss.mean() + anomaly_score.mean()

                    loss.backward()
                    trainer.optimizer.step()

                    loss_val = loss.item()
                    epoch_losses.append(loss_val)
                    numerical_batch_losses.append(loss_val)

        avg_numerical_loss = sum(numerical_batch_losses) / len(numerical_batch_losses) if numerical_batch_losses else 0.0
        logger.log_task_end("Task4数值训练", results={"avg_loss": avg_numerical_loss, "epochs": numerical_epochs})

        # 计算平均训练损失
        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        all_train_losses.append(avg_train_loss)

        # 获取当前学习率
        current_lr = trainer.optimizer.param_groups[0]['lr']

        # 记录epoch总结

        logger.log_epoch_summary(epoch, avg_train_loss, lr=current_lr,
                               image_loss=avg_image_loss, numerical_loss=avg_numerical_loss)

        # 每个epoch结束都进行验证集评估
        logger.log_info(f"Epoch {epoch + 1} 验证评估开始...")

        # 评估Task2：图像异常检测
        task2_val_auc = 0.0
        if image_val_loader:
            image_val_scores, image_val_labels = trainer.evaluate(image_val_loader)
            if len(image_val_scores) > 0:
                image_val_auc = roc_auc_score(image_val_labels, image_val_scores)
                task2_val_auc = image_val_auc
                image_val_threshold = np.percentile(image_val_scores, 95.0)
                image_val_predictions = (image_val_scores > image_val_threshold).astype(int)
                image_val_precision, image_val_recall, image_val_f1, _ = precision_recall_fscore_support(image_val_labels, image_val_predictions, average='binary')
                image_val_accuracy = accuracy_score(image_val_labels, image_val_predictions)

                logger.log_info(f"Task2验证 - AUC: {image_val_auc:.4f}, Precision: {image_val_precision:.4f}, Recall: {image_val_recall:.4f}, F1: {image_val_f1:.4f}, Acc: {image_val_accuracy:.4f}")

        # 评估Task4：数值异常检测
        task4_val_auc = 0.0
        if numerical_val_loader:
            numerical_val_scores, numerical_val_labels = trainer.evaluate(numerical_val_loader)
            if len(numerical_val_scores) > 0:
                numerical_val_auc = roc_auc_score(numerical_val_labels, numerical_val_scores)
                task4_val_auc = numerical_val_auc
                numerical_val_threshold = np.percentile(numerical_val_scores, 95.0)
                numerical_val_predictions = (numerical_val_scores > numerical_val_threshold).astype(int)
                numerical_val_precision, numerical_val_recall, numerical_val_f1, _ = precision_recall_fscore_support(numerical_val_labels, numerical_val_predictions, average='binary')
                numerical_val_accuracy = accuracy_score(numerical_val_labels, numerical_val_predictions)

                logger.log_info(f"Task4验证 - AUC: {numerical_val_auc:.4f}, Precision: {numerical_val_precision:.4f}, Recall: {numerical_val_recall:.4f}, F1: {numerical_val_f1:.4f}, Acc: {numerical_val_accuracy:.4f}")

        # 检查是否为最佳模型并保存
        current_combined_auc = (task2_val_auc + task4_val_auc) / 2

        if current_combined_auc > best_combined_auc:
            best_task2_auc = task2_val_auc
            best_task4_auc = task4_val_auc
            best_combined_auc = current_combined_auc
            best_epoch = epoch + 1

            # 保存最佳模型
            best_model_path = os.path.join(save_dir, 'best_model_checkpoint.pth')
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_task2_auc': best_task2_auc,
                'best_task4_auc': best_task4_auc,
                'best_combined_auc': best_combined_auc,
                'config': trainer.model_config if hasattr(trainer, 'model_config') else None
            }, best_model_path)

            logger.log_info(f"💾 保存最佳模型 - Epoch {epoch + 1}, 组合AUC: {best_combined_auc:.4f} (Task2: {best_task2_auc:.4f}, Task4: {best_task4_auc:.4f})")

        logger.log_info(f"Epoch {epoch + 1} 验证评估完成")

        # 学习率调度
        trainer.scheduler.step()

        # 更新学习率显示
        new_lr = trainer.optimizer.param_groups[0]['lr']
        if abs(new_lr - current_lr) > 1e-8:
            logger.log_info(".6f")

    # 记录最终结果
    final_results = {
        'final_train_loss': all_train_losses[-1] if all_train_losses else 0.0,
        'total_epochs': num_epochs,
        'best_epoch': best_epoch,
        'best_task2_auc': best_task2_auc,
        'best_task4_auc': best_task4_auc,
        'best_combined_auc': best_combined_auc
    }
    logger.log_task_end("整体训练", results=final_results)
    logger.close()

    print("分别训练完成！详细日志已保存。")
    return all_train_losses, []


def evaluate_separate_tasks(trainer, image_test_loader, numerical_test_loader, visualizer, threshold_percentile=95.0):
    """分别评估两种任务"""
    print("分别评估Task2和Task4...")

    # 设置评估日志
    eval_logger = setup_experiment_logging("results", "evaluation")

    results = {}

    # 评估Task2：图像异常检测
    eval_logger.log_task_start("Task2图像评估", f"测试样本数: {len(image_test_loader.dataset) if image_test_loader else 0}")
    print("评估Task2（图像异常检测）...")
    image_scores, image_labels = trainer.evaluate(image_test_loader)

    if len(image_scores) > 0:
        image_auc = roc_auc_score(image_labels, image_scores)
        image_threshold = np.percentile(image_scores, threshold_percentile)
        image_predictions = (image_scores > image_threshold).astype(int)
        image_precision, image_recall, image_f1, _ = precision_recall_fscore_support(image_labels, image_predictions, average='binary')
        image_accuracy = accuracy_score(image_labels, image_predictions)

        task2_metrics = {
            'auc': image_auc,
            'precision': image_precision,
            'recall': image_recall,
            'f1': image_f1,
            'accuracy': image_accuracy,
            'threshold': image_threshold
        }

        results['task2'] = {
            **task2_metrics,
            'scores': image_scores,
            'labels': image_labels,
            'predictions': image_predictions
        }

        # 记录评估结果到日志
        eval_logger.log_evaluation_results("Task2图像异常检测", task2_metrics)

        # 可视化Task2结果
        visualizer.plot_anomaly_scores(image_scores, image_labels, title="Task2: 图像异常检测结果", save_name="task2_anomaly_scores.png")
        visualizer.plot_roc_curve(image_labels, image_scores, title="Task2: 图像异常检测ROC曲线", save_name="task2_roc_curve.png")

    eval_logger.log_task_end("Task2图像评估")

    # 评估Task4：数值异常检测
    eval_logger.log_task_start("Task4数值评估", f"测试样本数: {len(numerical_test_loader.dataset) if numerical_test_loader else 0}")
    print("评估Task4（数值异常检测）...")
    numerical_scores, numerical_labels = trainer.evaluate(numerical_test_loader)

    if len(numerical_scores) > 0:
        numerical_auc = roc_auc_score(numerical_labels, numerical_scores)

        numerical_threshold = np.percentile(numerical_scores, threshold_percentile)
        numerical_predictions = (numerical_scores > numerical_threshold).astype(int)
        numerical_precision, numerical_recall, numerical_f1, _ = precision_recall_fscore_support(numerical_labels, numerical_predictions, average='binary')
        numerical_accuracy = accuracy_score(numerical_labels, numerical_predictions)

        task4_metrics = {
            'auc': numerical_auc,
            'precision': numerical_precision,
            'recall': numerical_recall,
            'f1': numerical_f1,
            'accuracy': numerical_accuracy,
            'threshold': numerical_threshold
        }

        results['task4'] = {
            **task4_metrics,
            'scores': numerical_scores,
            'labels': numerical_labels,
            'predictions': numerical_predictions
        }

        # 记录评估结果到日志
        eval_logger.log_evaluation_results("Task4数值异常检测", task4_metrics)

        # 可视化Task4结果
        visualizer.plot_anomaly_scores(numerical_scores, numerical_labels, title="Task4: 数值异常检测结果", save_name="task4_anomaly_scores.png")
        visualizer.plot_roc_curve(numerical_labels, numerical_scores, title="Task4: 数值异常检测ROC曲线", save_name="task4_roc_curve.png")

    eval_logger.log_task_end("Task4数值评估")
    eval_logger.close()

    return results


def create_separate_dataloaders(image_train_paths, image_test_paths, image_test_labels,
                               numerical_train_data, numerical_test_data, numerical_test_labels,
                               batch_size=8, val_split=0.2):
    """创建分别的数据加载器

    注意：对于异常检测任务，验证集从测试集中划分，因为训练集只包含正常样本
    """
    from torchvision import transforms
    from sklearn.model_selection import train_test_split

    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Task2数据加载器（图像）
    image_train_loader = None
    image_val_loader = None
    image_test_loader = None

    if image_train_paths:
        # 训练集：正常样本
        image_train_dataset = MultiModalDataset(
            image_paths=image_train_paths,
            transform=transform
        )
        image_train_loader = DataLoader(image_train_dataset, batch_size=batch_size, shuffle=True)

    if image_test_paths and image_test_labels is not None:
        # 从测试集中划分验证集和测试集
        # 确保验证集包含正常样本和异常样本
        test_indices = list(range(len(image_test_paths)))

        if len(test_indices) > 1:
            # 分层划分，确保验证集中包含两种类别
            normal_indices = [i for i, label in enumerate(image_test_labels) if label == 0]
            anomaly_indices = [i for i, label in enumerate(image_test_labels) if label == 1]

            # 为每类分别划分验证集
            val_normal_indices = normal_indices[:max(1, int(len(normal_indices) * val_split))]
            val_anomaly_indices = anomaly_indices[:max(1, int(len(anomaly_indices) * val_split))]

            val_indices = val_normal_indices + val_anomaly_indices
            test_indices = [i for i in test_indices if i not in val_indices]

            # 创建验证集
            image_val_paths = [image_test_paths[i] for i in val_indices]
            image_val_labels = [image_test_labels[i] for i in val_indices]

            # 创建测试集
            image_test_paths_final = [image_test_paths[i] for i in test_indices]
            image_test_labels_final = [image_test_labels[i] for i in test_indices]
        else:
            # 数据太少，直接使用全部作为测试集，验证集为空
            image_val_paths = []
            image_val_labels = []
            image_test_paths_final = image_test_paths
            image_test_labels_final = image_test_labels

        # 创建验证集加载器
        if image_val_paths:
            image_val_dataset = MultiModalDataset(
                image_paths=image_val_paths,
                transform=transform
            )
            image_val_dataset.labels = image_val_labels
            image_val_loader = DataLoader(image_val_dataset, batch_size=batch_size, shuffle=False)

        # 创建测试集加载器
        if image_test_paths_final:
            image_test_dataset = MultiModalDataset(
                image_paths=image_test_paths_final,
                transform=transform
            )
            image_test_dataset.labels = image_test_labels_final
            image_test_loader = DataLoader(image_test_dataset, batch_size=batch_size, shuffle=False)

    # Task4数据加载器（数值）
    numerical_train_loader = None
    numerical_val_loader = None
    numerical_test_loader = None

    if numerical_train_data is not None:
        # 训练集：正常样本
        numerical_train_dataset = MultiModalDataset(
            numerical_data=numerical_train_data
        )
        numerical_train_loader = DataLoader(numerical_train_dataset, batch_size=batch_size, shuffle=True)

    if numerical_test_data is not None and numerical_test_labels is not None:
        # 从测试集中划分验证集和测试集
        test_indices = list(range(len(numerical_test_data)))

        if len(test_indices) > 1:
            # 分层划分，确保验证集中包含两种类别
            normal_indices = [i for i, label in enumerate(numerical_test_labels) if label == 0]
            anomaly_indices = [i for i, label in enumerate(numerical_test_labels) if label == 1]

            # 为每类分别划分验证集
            val_normal_indices = normal_indices[:max(1, int(len(normal_indices) * val_split))]
            val_anomaly_indices = anomaly_indices[:max(1, int(len(anomaly_indices) * val_split))]

            val_indices = val_normal_indices + val_anomaly_indices
            test_indices = [i for i in test_indices if i not in val_indices]

            # 创建验证集
            numerical_val_data = numerical_test_data[val_indices]
            numerical_val_labels = numerical_test_labels[val_indices]

            # 创建测试集
            test_indices_array = np.array(test_indices)
            numerical_test_data_final = numerical_test_data[test_indices_array]
            numerical_test_labels_final = numerical_test_labels[test_indices_array]
        else:
            # 数据太少，直接使用全部作为测试集，验证集为空
            numerical_val_data = None
            numerical_val_labels = None
            numerical_test_data_final = numerical_test_data
            numerical_test_labels_final = numerical_test_labels

        # 创建验证集加载器
        if numerical_val_data is not None and len(numerical_val_data) > 0:
            numerical_val_dataset = MultiModalDataset(
                numerical_data=numerical_val_data
            )
            numerical_val_dataset.labels = numerical_val_labels
            numerical_val_loader = DataLoader(numerical_val_dataset, batch_size=batch_size, shuffle=False)

        # 创建测试集加载器
        if numerical_test_data_final is not None and len(numerical_test_data_final) > 0:
            numerical_test_dataset = MultiModalDataset(
                numerical_data=numerical_test_data_final
            )
            numerical_test_dataset.labels = numerical_test_labels_final
            numerical_test_loader = DataLoader(numerical_test_dataset, batch_size=batch_size, shuffle=False)

    return (image_train_loader, image_val_loader, image_test_loader,
            numerical_train_loader, numerical_val_loader, numerical_test_loader)


def main():
    """主函数"""
    print("多模态异常检测Transformer演示")
    print("=" * 50)
    print("使用策略：同一个模型分别训练Task2和Task4")
    print("=" * 50)

    # 解析参数
    args = parse_args()

    # 获取配置
    if args.config == 'small':
        model_config, training_config, eval_config = get_small_config()
    elif args.config == 'large':
        model_config, training_config, eval_config = get_large_config()
    else:
        model_config, training_config, eval_config = get_config()

    # 覆盖配置
    if args.epochs:
        training_config.num_epochs = args.epochs
    if args.batch_size:
        training_config.batch_size = args.batch_size
    if args.quick_test:
        training_config.num_epochs = 3
        model_config.depth = 2

    # 设置设备
    device = setup_device(args.device)

    # 创建可视化器
    visualizer = AnomalyVisualizer(args.save_dir)

    try:
        # 加载数据
        data = load_data(training_config)

        # 创建分别的数据加载器
        (image_train_loader, image_val_loader, image_test_loader,
         numerical_train_loader, numerical_val_loader, numerical_test_loader) = create_separate_dataloaders(
            *data, batch_size=training_config.batch_size
        )

        # 创建模型和训练器
        model, trainer = create_model_and_trainer(model_config, training_config, device)

        # 如果指定了checkpoint，加载并继续训练
        if args.resume:
            print(f"Loading checkpoint from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

            # 如果checkpoint包含优化器状态，也加载
            if 'optimizer_state_dict' in checkpoint:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded")

            # 如果checkpoint包含epoch信息，更新训练轮数
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                remaining_epochs = training_config.num_epochs - start_epoch
                if remaining_epochs > 0:
                    training_config.num_epochs = remaining_epochs
                    print(f"Resuming from epoch {start_epoch}, remaining {remaining_epochs} epochs")
                else:
                    print(f"Model already trained for {training_config.num_epochs} epochs, finishing training")

            print("Checkpoint loaded successfully")
            print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # 分别训练两种任务
        train_losses, _ = train_task_separately(
            trainer, image_train_loader, numerical_train_loader,
            image_val_loader, numerical_val_loader,
            training_config.num_epochs, args.save_dir
        )

        # 绘制训练历史
        visualizer.plot_training_history(train_losses, [])

        # 分别评估两种任务
        task_results = evaluate_separate_tasks(
            trainer, image_test_loader, numerical_test_loader,
            visualizer, eval_config.anomaly_threshold_percentile
        )

        # 输出总结结果
        print("\n训练结果总结:")
        print("=" * 30)

        if 'task2' in task_results:
            print("Task2（图像异常检测）:")


        if 'task4' in task_results:
            print("\nTask4（数值异常检测）:")

        # 创建评估报告
        report_metrics = {
            'depth': model_config.depth,
            'embed_dim': model_config.embed_dim,
            'epochs': training_config.num_epochs,
            'total_train_loss': train_losses[-1] if train_losses else 0.0
        }

        # 计算confusion matrix等额外信息
        def calculate_confusion_matrix(labels, predictions):
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(labels, predictions)
            return cm

        # 添加评估结果
        if 'task2' in task_results and task_results['task2']:
            task2_data = task_results['task2']
            report_metrics.update({
                'task2_auc': task2_data.get('auc', 'N/A'),
                'task2_precision': task2_data.get('precision', 'N/A'),
                'task2_recall': task2_data.get('recall', 'N/A'),
                'task2_f1': task2_data.get('f1', 'N/A'),
                'task2_accuracy': task2_data.get('accuracy', 'N/A'),
                'task2_threshold': task2_data.get('threshold', 'N/A'),
                'task2_confusion_matrix': calculate_confusion_matrix(task2_data.get('labels', []), task2_data.get('predictions', [])) if 'labels' in task2_data and 'predictions' in task2_data else None
            })

        if 'task4' in task_results and task_results['task4']:
            task4_data = task_results['task4']
            report_metrics.update({
                'task4_auc': task4_data.get('auc', 'N/A'),
                'task4_precision': task4_data.get('precision', 'N/A'),
                'task4_recall': task4_data.get('recall', 'N/A'),
                'task4_f1': task4_data.get('f1', 'N/A'),
                'task4_accuracy': task4_data.get('accuracy', 'N/A'),
                'task4_threshold': task4_data.get('threshold', 'N/A'),
                'task4_confusion_matrix': calculate_confusion_matrix(task4_data.get('labels', []), task4_data.get('predictions', [])) if 'labels' in task4_data and 'predictions' in task4_data else None
            })

        # 添加数据集信息
        image_train_count = len(image_train_loader.dataset) if image_train_loader else 0
        image_test_count = len(image_test_loader.dataset) if image_test_loader else 0
        numerical_train_count = len(numerical_train_loader.dataset) if numerical_train_loader else 0
        numerical_test_count = len(numerical_test_loader.dataset) if numerical_test_loader else 0

        report_metrics.update({
            'image_train_samples': image_train_count,
            'image_test_samples': image_test_count,
            'numerical_train_samples': numerical_train_count,
            'numerical_test_samples': numerical_test_count,
            'total_train_samples': image_train_count + numerical_train_count,
            'total_test_samples': image_test_count + numerical_test_count
        })
        visualizer.create_summary_report(report_metrics)

        # 保存结果
        results = {
            'model': model,
            'task_results': task_results,
            'config': model_config
        }
        save_results(results, training_config, args.save_dir)

        print("\n演示完成！")
        print(f"结果保存在: {args.save_dir}")
        print("最佳模型已保存为: best_model_checkpoint.pth")

        # 检查最佳模型是否存在并显示信息
        best_model_path = os.path.join(args.save_dir, 'best_model_checkpoint.pth')
        if os.path.exists(best_model_path):
            best_checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
            print(f"最佳模型信息:")
            print(f"   - 最佳epoch: {best_checkpoint.get('epoch', 'N/A')}")
            print(f"   - Task2 AUC: {best_checkpoint.get('best_task2_auc', 'N/A'):.4f}")
            print(f"   - Task4 AUC: {best_checkpoint.get('best_task4_auc', 'N/A'):.4f}")
            print(f"   - 组合AUC: {best_checkpoint.get('best_combined_auc', 'N/A'):.4f}")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()