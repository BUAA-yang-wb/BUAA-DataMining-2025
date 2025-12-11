import os
import torch

# 路径配置
# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据集根目录
DATA_ROOT = os.path.join(BASE_DIR, "Image_Anomaly_Detection")
LABEL_FILE = os.path.join(DATA_ROOT, "image_anomaly_labels.json")

# 类别
CATEGORIES = ['hazelnut', 'zipper']

# 训练超参数
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 100  # 增加训练轮次，给复杂纹理更多学习时间
LEARNING_RATE = 1e-3

# 模型配置
USE_RESNET = True  # True: 使用预训练ResNet18编码器 (推荐), False: 使用原始自动编码器
USE_SSIM_LOSS = True  # True: 使用SSIM+MSE组合损失, False: 仅使用MSE
SSIM_WEIGHT = 0.5  # SSIM损失的权重 (MSE权重 = 1 - SSIM_WEIGHT)

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"Configuration loaded. Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print(f"Configuration loaded. Device: {DEVICE} (GPU not available, using CPU)")
print(f"Data Root: {DATA_ROOT}")
