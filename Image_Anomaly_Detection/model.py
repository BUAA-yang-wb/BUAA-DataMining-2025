import torch
import torch.nn as nn
import torchvision.models as models

class Autoencoder(nn.Module):
    """原始的自动编码器模型（保留用于兼容性）"""
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Enhanced Encoder with Batch Normalization and deeper layers
        # 输入: [Batch, 3, 128, 128]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),  # -> [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # -> [B, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # -> [B, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1), # -> [B, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # 额外一层压缩，提高特征抽象能力
            nn.Conv2d(512, 512, 3, stride=2, padding=1), # -> [B, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Enhanced Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1), # -> [B, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1), # -> [B, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # -> [B, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # -> [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1), # -> [B, 3, 128, 128]
            nn.Sigmoid() # 输出范围 [0, 1]
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class ResNetAutoencoder(nn.Module):
    """使用预训练 ResNet18 作为编码器的自动编码器"""
    def __init__(self, pretrained=True, freeze_encoder_layers=True):
        super(ResNetAutoencoder, self).__init__()
        
        # 使用预训练的 ResNet18 作为编码器
        # 使用新的 weights API 避免 deprecated 警告
        if pretrained:
            from torchvision.models import ResNet18_Weights
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        
        # 提取 ResNet18 的特征提取部分（去掉全连接层和池化层）
        # ResNet18 输出: [B, 512, 4, 4] (对于 128x128 输入)
        self.encoder = nn.Sequential(
            resnet.conv1,      # [B, 64, 64, 64]
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,    # [B, 64, 32, 32]
            resnet.layer1,     # [B, 64, 32, 32]
            resnet.layer2,     # [B, 128, 16, 16]
            resnet.layer3,     # [B, 256, 8, 8]
            resnet.layer4      # [B, 512, 4, 4]
        )
        
        # 冻结编码器的前几层，只微调后面的层
        if freeze_encoder_layers and pretrained:
            # 冻结 conv1, bn1, layer1 的参数
            for param in resnet.conv1.parameters():
                param.requires_grad = False
            for param in resnet.bn1.parameters():
                param.requires_grad = False
            for param in resnet.layer1.parameters():
                param.requires_grad = False
            print("  → Encoder: First layers frozen, layer2-4 trainable")
        else:
            print("  → Encoder: All layers trainable")
        
        # 解码器：使用转置卷积逐步上采样
        self.decoder = nn.Sequential(
            # [B, 512, 4, 4] -> [B, 256, 8, 8]
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # [B, 256, 8, 8] -> [B, 128, 16, 16]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # [B, 128, 16, 16] -> [B, 64, 32, 32]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # [B, 64, 32, 32] -> [B, 32, 64, 64]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # [B, 32, 64, 64] -> [B, 3, 128, 128]
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
    
    def forward(self, x):
        # 编码
        features = self.encoder(x)
        # 解码
        reconstruction = self.decoder(features)
        return reconstruction
