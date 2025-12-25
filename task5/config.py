"""
配置文件：多模态异常检测Transformer的超参数配置
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """模型配置"""
    # 图像相关
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3

    # 数值特征
    num_numerical_features: int = 6

    # Transformer配置
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0

    # 训练配置
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 16
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10

    # 数据集配置
    image_category: str = "hazelnut"  # 或 "zipper"
    train_normal_only: bool = True

    # 路径配置
    image_data_path: str = "Image_Anomaly_Detection"
    thyroid_data_path: str = "thyroid"

    # 设备配置
    device: str = "auto"  # "auto", "cuda", "cpu"


@dataclass
class EvaluationConfig:
    """评估配置"""
    anomaly_threshold_percentile: float = 95.0
    save_results: bool = True
    results_dir: str = "results"

    # 指标配置
    compute_auc: bool = True
    compute_precision_recall: bool = True
    compute_f1: bool = True


def get_config() -> tuple[ModelConfig, TrainingConfig, EvaluationConfig]:
    """获取默认配置"""
    model_config = ModelConfig()
    training_config = TrainingConfig()
    eval_config = EvaluationConfig()

    return model_config, training_config, eval_config


def get_small_config() -> tuple[ModelConfig, TrainingConfig, EvaluationConfig]:
    """获取轻量级配置（用于快速测试）"""
    model_config = ModelConfig(
        embed_dim=192,
        depth=4,
        num_heads=6,
    )

    training_config = TrainingConfig(
        batch_size=8,
        num_epochs=5,
        learning_rate=5e-4,
    )

    eval_config = EvaluationConfig()

    return model_config, training_config, eval_config


def get_large_config() -> tuple[ModelConfig, TrainingConfig, EvaluationConfig]:
    """获取大型配置（更高性能）"""
    model_config = ModelConfig(
        embed_dim=768,
        depth=12,
        num_heads=12,
    )

    training_config = TrainingConfig(
        batch_size=32,
        num_epochs=100,
        learning_rate=5e-5,
    )

    eval_config = EvaluationConfig()

    return model_config, training_config, eval_config
