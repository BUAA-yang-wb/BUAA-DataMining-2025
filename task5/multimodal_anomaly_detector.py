"""
统一多模态异常检测Transformer架构
支持图像异常检测和数值数据异常检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pandas as pd
import json
import os
from typing import Optional, Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


class MultiModalPatchEmbed(nn.Module):
    """多模态Patch嵌入层，支持图像和数值数据"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_numerical_features=6):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 图像patch嵌入
        self.image_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 数值特征嵌入
        self.numerical_proj = nn.Linear(num_numerical_features, embed_dim)

        # 模态类型嵌入
        self.modality_embed = nn.Embedding(2, embed_dim)  # 0: image, 1: numerical

        # 位置嵌入 - 创建足够长的位置嵌入以适应不同序列长度
        max_seq_len = self.num_patches + 10  # 留出余量
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.initialize_weights()

    def initialize_weights(self):
        # 初始化权重
        torch.nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_image(self, x):
        """处理图像数据"""
        B = x.shape[0]
        x = self.image_proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # 添加模态嵌入
        modality_tokens = self.modality_embed(torch.zeros(B, self.num_patches, dtype=torch.long, device=x.device))
        x = x + modality_tokens

        return x

    def forward_numerical(self, x):
        """处理数值数据"""
        B = x.shape[0]
        x = self.numerical_proj(x)  # (B, embed_dim)
        x = x.unsqueeze(1)  # (B, 1, embed_dim)

        # 添加模态嵌入
        modality_tokens = self.modality_embed(torch.ones(B, 1, dtype=torch.long, device=x.device))
        x = x + modality_tokens

        return x

    def forward(self, image_data=None, numerical_data=None):
        """统一前向传播"""
        embeddings = []

        if image_data is not None:
            img_embeds = self.forward_image(image_data)
            embeddings.append(img_embeds)

        if numerical_data is not None:
            num_embeds = self.forward_numerical(numerical_data)
            embeddings.append(num_embeds)

        if len(embeddings) == 0:
            raise ValueError("至少需要提供一种模态的数据")

        # 拼接不同模态的embeddings
        x = torch.cat(embeddings, dim=1)  # (B, total_patches, embed_dim)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加位置嵌入 - 根据实际序列长度截取
        seq_len = x.shape[1]
        pos_embed = self.absolute_pos_embed[:, :seq_len, :]
        x = x + pos_embed

        return x


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossModalAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()  # 暂时不使用drop path
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AnomalyDetectionHead(nn.Module):
    """异常检测头"""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # 重构分支 - 动态生成重构目标
        self.reconstruction_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # 异常分数计算
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, seq_len, embed_dim) - seq_len 包括 CLS token
        cls_token = x[:, 0]  # (B, embed_dim)
        patch_tokens = x[:, 1:]  # (B, seq_len-1, embed_dim)

        # 使用CLS token预测patch tokens的重构
        # 这里简化处理：使用CLS token对每个patch位置进行预测
        batch_size, seq_len, embed_dim = patch_tokens.shape

        # 扩展CLS token到所有patch位置
        cls_expanded = cls_token.unsqueeze(1).expand(-1, seq_len, -1)  # (B, seq_len, embed_dim)

        # 重构预测
        reconstructed = self.reconstruction_encoder(cls_expanded)  # (B, seq_len, embed_dim)

        # 计算重构误差
        reconstruction_loss = F.mse_loss(reconstructed, patch_tokens, reduction='none')
        reconstruction_loss = reconstruction_loss.mean(dim=(1, 2))  # (B,)

        # 异常分数 - 基于CLS token
        anomaly_score = self.anomaly_scorer(cls_token).squeeze(-1)  # (B,)

        return reconstruction_loss, anomaly_score, reconstructed


class MultiModalAnomalyTransformer(nn.Module):
    """统一多模态异常检测Transformer"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_numerical_features=6,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0.):
        super().__init__()

        self.num_numerical_features = num_numerical_features
        self.embed_dim = embed_dim

        # Patch嵌入
        self.patch_embed = MultiModalPatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, num_numerical_features=num_numerical_features
        )
        num_patches = self.patch_embed.num_patches

        # 位置嵌入会在patch_embed中处理
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=nn.LayerNorm)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 异常检测头
        self.anomaly_head = AnomalyDetectionHead(embed_dim)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, image_data=None, numerical_data=None):
        # 嵌入
        x = self.patch_embed(image_data, numerical_data)  # (B, num_patches + 1, embed_dim)
        x = self.pos_drop(x)

        # Transformer编码器
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # 异常检测
        reconstruction_loss, anomaly_score, reconstructed = self.anomaly_head(x)

        return reconstruction_loss, anomaly_score, reconstructed, x


class MultiModalDataset(Dataset):
    """多模态数据集"""

    def __init__(self, image_paths=None, numerical_data=None, labels=None, transform=None):
        self.image_paths = image_paths or []
        self.numerical_data = numerical_data
        self.labels = labels
        self.transform = transform

        # 确定数据集大小
        if self.image_paths:
            self.length = len(self.image_paths)
        elif self.numerical_data is not None:
            self.length = len(self.numerical_data)
        else:
            self.length = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {}

        # 处理图像数据
        if self.image_paths:
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            sample['image'] = image

        # 处理数值数据
        if self.numerical_data is not None:
            sample['numerical'] = torch.tensor(self.numerical_data[idx], dtype=torch.float32)

        # 处理标签
        if self.labels is not None:
            sample['label'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return sample


class Trainer:
    """训练器"""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            self.optimizer.zero_grad()

            # 准备数据
            image_data = batch.get('image')
            numerical_data = batch.get('numerical')

            if image_data is not None:
                image_data = image_data.to(self.device)
            if numerical_data is not None:
                numerical_data = numerical_data.to(self.device)

            # 前向传播
            reconstruction_loss, anomaly_score, _, _ = self.model(image_data, numerical_data)

            # 计算总损失（最小化正常样本的重构误差和异常分数）
            loss = reconstruction_loss.mean() + anomaly_score.mean()

            # 反向传播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估进度", leave=False):
                # 准备数据
                image_data = batch.get('image')
                numerical_data = batch.get('numerical')
                labels = batch.get('label', torch.zeros(len(batch['image'] if image_data is not None else batch['numerical'])))

                if image_data is not None:
                    image_data = image_data.to(self.device)
                if numerical_data is not None:
                    numerical_data = numerical_data.to(self.device)

                # 前向传播
                reconstruction_loss, anomaly_score, _, _ = self.model(image_data, numerical_data)

                # 组合异常分数
                final_score = reconstruction_loss + anomaly_score

                all_scores.extend(final_score.cpu().numpy())
                all_labels.extend(labels.numpy())

        return np.array(all_scores), np.array(all_labels)


def load_image_anomaly_data(base_path, category, train_normal_only=True):
    """加载图像异常检测数据"""
    train_good_paths = []
    train_bad_paths = []
    test_paths = []
    test_labels = []

    # 训练集
    train_good_dir = os.path.join(base_path, category, 'train', 'good')
    if os.path.exists(train_good_dir):
        train_good_paths = [os.path.join(train_good_dir, f) for f in os.listdir(train_good_dir) if f.endswith('.png')]

    if not train_normal_only:
        train_bad_dir = os.path.join(base_path, category, 'train', 'bad')
        if os.path.exists(train_bad_dir):
            train_bad_paths = [os.path.join(train_bad_dir, f) for f in os.listdir(train_bad_dir) if f.endswith('.png')]

    # 测试集
    test_dir = os.path.join(base_path, category, 'test')
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
        test_paths = [os.path.join(test_dir, f) for f in test_files]

    # 加载标签
    labels_file = os.path.join(base_path, 'image_anomaly_labels.json')
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)

        for path in test_paths:
            relative_path = os.path.relpath(path, base_path).replace('\\', '/')
            if relative_path in labels_data:
                label = 1 if labels_data[relative_path]['label'] == 'bad' else 0
                test_labels.append(label)
            else:
                test_labels.append(0)  # 默认正常

    train_paths = train_good_paths + train_bad_paths if not train_normal_only else train_good_paths

    return train_paths, test_paths, test_labels


def load_thyroid_data(base_path):
    """加载甲状腺疾病数据"""
    train_file = os.path.join(base_path, 'train-set.csv')
    test_file = os.path.join(base_path, 'test-set.csv')

    # 训练集（只包含正常样本）
    train_df = pd.read_csv(train_file)
    train_features = train_df[['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6']].values

    # 测试集
    test_df = pd.read_csv(test_file)
    test_features = test_df[['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6']].values
    test_labels = test_df['label'].values

    return train_features, test_features, test_labels


def create_dataloaders(image_train_paths, image_test_paths, image_test_labels,
                      numerical_train_data, numerical_test_data, numerical_test_labels,
                      batch_size=32):
    """创建数据加载器"""

    from torchvision import transforms

    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = MultiModalDataset(
        image_paths=image_train_paths,
        numerical_data=numerical_train_data,
        transform=transform
    )

    test_dataset = MultiModalDataset(
        image_paths=image_test_paths,
        numerical_data=numerical_test_data,
        transform=transform
    )

    # 添加标签到测试数据集
    test_dataset.labels = image_test_labels if image_test_paths else numerical_test_labels

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def main():
    """主函数：演示统一异常检测"""

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("加载数据...")

    # 图像数据（只用hazelnut的正常样本训练）
    image_train_paths, image_test_paths, image_test_labels = load_image_anomaly_data(
        'Image_Anomaly_Detection', 'hazelnut', train_normal_only=True
    )

    # 数值数据
    numerical_train_data, numerical_test_data, numerical_test_labels = load_thyroid_data('thyroid')

    print(f"图像训练样本: {len(image_train_paths)}")
    print(f"图像测试样本: {len(image_test_paths)}")
    print(f"数值训练样本: {len(numerical_train_data)}")
    print(f"数值测试样本: {len(numerical_test_data)}")

    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(
        image_train_paths, image_test_paths, image_test_labels,
        numerical_train_data, numerical_test_data, numerical_test_labels,
        batch_size=16
    )

    # 创建模型
    model = MultiModalAnomalyTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=6, num_heads=6,
        num_numerical_features=6
    )

    # 创建训练器
    trainer = Trainer(model, device)

    # 训练
    print("开始训练...")
    num_epochs = 10

    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        print(".4f")

        # 定期评估
        if (epoch + 1) % 5 == 0:
            scores, labels = trainer.evaluate(test_loader)
            auc = roc_auc_score(labels, scores)
            print(".4f")

    # 最终评估
    print("\n最终评估:")
    scores, labels = trainer.evaluate(test_loader)

    auc = roc_auc_score(labels, scores)
    print(".4f")

    # 计算其他指标
    threshold = np.percentile(scores, 95)  # 95%分位数作为阈值
    predictions = (scores > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    print(".4f")
    print(".4f")
    print(".4f")

    print("\n训练完成！")


if __name__ == "__main__":
    main()
