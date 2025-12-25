#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM时间序列预测模型
用于预测室外温度(OT)基于过去2小时的气象数据
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts, ExponentialDecay
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import logging
import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(data_dir='data'):
    """
    加载数据并进行标准化处理

    Args:
        data_dir (str): 数据文件夹路径

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    logger = logging.getLogger()
    logger.info("=== Loading and Preprocessing Data ===")
    print("=== Loading and Preprocessing Data ===")

    # 加载数据
    train_data = np.load(os.path.join(data_dir, 'train_data.npz'))
    val_data = np.load(os.path.join(data_dir, 'val_data.npz'))
    test_data = np.load(os.path.join(data_dir, 'test_data.npz'))

    X_train = train_data['X']
    y_train = train_data['y']
    X_val = val_data['X']
    y_val = val_data['y']
    X_test = test_data['X']
    y_test = test_data['y']

    print("Original data shapes:")
    print("  X_train: {}".format(X_train.shape))
    print("  y_train: {}".format(y_train.shape))
    print("  X_val: {}".format(X_val.shape))
    print("  y_val: {}".format(y_val.shape))
    print("  X_test: {}".format(X_test.shape))
    print("  y_test: {}".format(y_test.shape))

    # 数据标准化 - 保持时间序列结构
    print("\nStandardizing data (preserving temporal structure)...")

    n_samples_train, n_timesteps, n_features = X_train.shape
    n_samples_val = X_val.shape[0]
    n_samples_test = X_test.shape[0]

    # 创建标准化参数存储
    feature_means = np.zeros(n_features)
    feature_stds = np.zeros(n_features)

    # 对每个特征分别进行标准化（保持时间结构）
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    for i in range(n_features):
        # 使用训练集计算每个特征的均值和标准差
        feature_mean = X_train[:, :, i].mean()
        feature_std = X_train[:, :, i].std()

        feature_means[i] = feature_mean
        feature_stds[i] = feature_std

        # 标准化训练集
        X_train_scaled[:, :, i] = (X_train[:, :, i] - feature_mean) / (feature_std + 1e-8)

        # 使用训练集的统计量标准化验证集和测试集
        X_val_scaled[:, :, i] = (X_val[:, :, i] - feature_mean) / (feature_std + 1e-8)
        X_test_scaled[:, :, i] = (X_test[:, :, i] - feature_mean) / (feature_std + 1e-8)

    # 对目标变量进行标准化
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # 存储标准化参数
    scaler_params = {
        'feature_means': feature_means,
        'feature_stds': feature_stds,
        'y_mean': y_scaler.mean_[0],
        'y_scale': y_scaler.scale_[0]
    }

    print("After preprocessing:")
    print("  X_train: {} (standardized)".format(X_train_scaled.shape))
    print("  X_val: {} (standardized)".format(X_val_scaled.shape))
    print("  X_test: {} (standardized)".format(X_test_scaled.shape))
    print("  Feature means range: [{:.3f}, {:.3f}]".format(feature_means.min(), feature_means.max()))
    print("  Feature stds range: [{:.3f}, {:.3f}]".format(feature_stds.min(), feature_stds.max()))
    print("  Target (OT) mean: {:.3f}, std: {:.3f}".format(y_scaler.mean_[0], y_scaler.scale_[0]))

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler_params

def build_lstm_model(input_shape, lstm_units=96, dropout_rate=0.25):
    """
    构建增强型LSTM模型（增加复杂度）

    Args:
        input_shape (tuple): 输入形状 (timesteps, features)
        lstm_units (int): LSTM单元数
        dropout_rate (float): Dropout比率

    Returns:
        tf.keras.Model: 编译后的LSTM模型
    """
    logger = logging.getLogger()
    logger.info("=== Building Enhanced LSTM Model ===")
    logger.info("Input shape: {}".format(input_shape))
    logger.info("LSTM units: {}".format(lstm_units))
    logger.info("Dropout rate: {}".format(dropout_rate))

    print("\n=== Building Enhanced LSTM Model ===")
    print("Input shape: {}".format(input_shape))
    print("LSTM units: {}".format(lstm_units))
    print("Dropout rate: {}".format(dropout_rate))
    print("Architecture: 2-LSTM layers + balanced Dense layer (optimized complexity)")

    # 输入层
    inputs = tf.keras.Input(shape=input_shape)

    # LSTM层1 (基础特征提取) - 平衡复杂度
    x = LSTM(lstm_units,
             return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l2(0.003),  # 降低L2正则化
             recurrent_regularizer=tf.keras.regularizers.l2(0.003))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # LSTM层2 (高层特征抽象) - 平衡复杂度
    x = LSTM(lstm_units // 2,
             kernel_regularizer=tf.keras.regularizers.l2(0.003),
             recurrent_regularizer=tf.keras.regularizers.l2(0.003))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Dense层 - 平衡表示能力
    x = Dense(64, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.003))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate * 0.9)(x)

    # 最终输出层
    outputs = Dense(1, activation='linear',
                   kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

    # 构建模型
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # 编译模型
    optimizer = Adam(learning_rate=0.01, clipnorm=1.0)  # 增加学习率适应优化后的模型
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae', 'mse'])

    print("Enhanced LSTM Model summary:")
    print("Total parameters: {:,}".format(model.count_params()))
    print("Trainable parameters: {:,}".format(sum([layer.count_params() for layer in model.layers if len(layer.weights) > 0])))

    # 显示模型结构
    model.summary()

    return model

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, model, epochs=100, batch_size=32):
    """
    训练LSTM模型

    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        X_test, y_test: 测试数据
        model: LSTM模型
        epochs (int): 训练轮数
        batch_size (int): 批大小

    Returns:
        tf.keras.Model: 训练好的模型
    """
    logger = logging.getLogger()
    logger.info("=== Training LSTM Model ===")
    logger.info("Training samples: {}".format(len(X_train)))
    logger.info("Validation samples: {}".format(len(X_val)))
    logger.info("Test samples: {}".format(len(X_test)))
    logger.info("Epochs: {}".format(epochs))
    logger.info("Batch size: {}".format(batch_size))

    print("\n=== Training LSTM Model ===")
    print("Training samples: {}".format(len(X_train)))
    print("Validation samples: {}".format(len(X_val)))
    print("Test samples: {}".format(len(X_test)))
    print("Epochs: {}".format(epochs))
    print("Batch size: {}".format(batch_size))

    # 回调函数
    callbacks = [
        # 早停 - 减少耐心值，防止过拟合
        EarlyStopping(
            monitor='val_loss',
            patience=15,  # 从15减少到10，更早停止
            restore_best_weights=True,
            verbose=1
        ),

        # 学习率调度 - 优化版 ReduceLROnPlateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
            mode='min',
            cooldown=2    # 衰减后等待2个epoch再继续监控
        ),

        # 模型检查点
        ModelCheckpoint(
            'save/model.keras',  # 使用新的Keras格式
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # 自定义回调来记录每个epoch的结果到日志
    class LoggingCallback(tf.keras.callbacks.Callback):
        def __init__(self, logger):
            super().__init__()
            self.logger = logger

        def on_epoch_end(self, epoch, logs=None):
            if logs:
                self.logger.info("Epoch {:3d}: loss={:.4f}, val_loss={:.4f}, mae={:.4f}, val_mae={:.4f}".format(
                    epoch + 1,
                    logs.get('loss', 0),
                    logs.get('val_loss', 0),
                    logs.get('mae', 0),
                    logs.get('val_mae', 0)
                ))

    # 添加自定义日志回调
    logging_callback = LoggingCallback(logger)
    callbacks.append(logging_callback)

    # 训练模型 - 使用更保守的训练策略
    print("Using conservative training strategy to prevent overfitting...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1  # 仍然显示在控制台
    )

    return model, history

def evaluate_model(model, X_test, y_test, scaler_params):
    """
    评估模型性能

    Args:
        model: 训练好的模型
        X_test, y_test: 测试数据（已标准化）
        scaler_params: 标准化参数
    """
    logger = logging.getLogger()
    logger.info("=== Model Evaluation ===")
    print("\n=== Model Evaluation ===")

    # 预测（在标准化空间）
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # 调试信息：检查标准化空间的数据
    print("Debug - Standardized space check (first 5 samples):")
    print("  y_pred_scaled:", y_pred_scaled[:5])
    print("  y_test_scaled:", y_test[:5])
    print("  y_pred_scaled range: [{:.3f}, {:.3f}]".format(y_pred_scaled.min(), y_pred_scaled.max()))
    print("  y_test_scaled range: [{:.3f}, {:.3f}]".format(y_test.min(), y_test.max()))

    # 反标准化预测结果和真实值
    y_pred = y_pred_scaled * scaler_params['y_scale'] + scaler_params['y_mean']
    y_test_original = y_test * scaler_params['y_scale'] + scaler_params['y_mean']

    print("\nDebug - Original space check (first 5 samples):")
    print("  y_pred (original):", y_pred[:5])
    print("  y_test (original):", y_test_original[:5])
    print("  Used y_mean: {:.6f}, y_scale: {:.6f}".format(scaler_params['y_mean'], scaler_params['y_scale']))

    # 计算指标
    mae = mean_absolute_error(y_test_original, y_pred)
    mse = mean_squared_error(y_test_original, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred)

    logger.info("Test Results (original scale):")
    logger.info("  MAE (Mean Absolute Error): {:.4f}".format(mae))
    logger.info("  RMSE (Root Mean Squared Error): {:.4f}".format(rmse))
    logger.info("  R² (R-squared): {:.4f}".format(r2))

    print("\nFinal Test Results (original scale):")
    print("  MAE (Mean Absolute Error): {:.4f}".format(mae))
    print("  RMSE (Root Mean Squared Error): {:.4f}".format(rmse))
    print("  R² (R-squared): {:.4f}".format(r2))

    # 性能评估
    print("\nPerformance Assessment:")
    if mae < 1.0:
        print("  MAE excellent (< 1.0)")
    elif mae < 2.0:
        print("  MAE good (< 2.0)")
    else:
        print("  MAE needs improvement (> 2.0)")

    if r2 > 0.8:
        print("  R² excellent (> 0.8)")
    elif r2 > 0.6:
        print("  R² good (> 0.6)")
    else:
        print("  R² needs improvement (< 0.6)")

    return y_pred, mae, rmse, r2

def plot_training_history(history):
    """
    绘制训练历史

    Args:
        history: 训练历史对象
    """
    print("\n=== Plotting Training History ===")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True)

    # MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)

    # MSE
    ax3.plot(history.history['mse'], label='Training MSE')
    ax3.plot(history.history['val_mse'], label='Validation MSE')
    ax3.set_title('Model MSE')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MSE')
    ax3.legend()
    ax3.grid(True)

    # Learning Rate (if available)
    if 'lr' in history.history:
        ax4.plot(history.history['lr'], label='Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Training history plot saved as 'plots/training_history.png'")

def plot_predictions(y_test_original, y_pred_original, num_samples=200):
    """
    绘制预测结果对比（原始尺度）

    Args:
        y_test_original: 真实值（原始尺度）
        y_pred_original: 预测值（原始尺度）
        num_samples: 显示的样本数量
    """
    print("\n=== Plotting Predictions ===")

    plt.figure(figsize=(15, 8))

    # 选择要显示的样本范围
    start_idx = 0
    end_idx = min(num_samples, len(y_test_original))

    # 绘制预测vs实际
    plt.subplot(2, 1, 1)
    plt.plot(y_test_original[start_idx:end_idx], label='Actual OT', color='blue', alpha=0.7)
    plt.plot(y_pred_original[start_idx:end_idx], label='Predicted OT', color='red', alpha=0.7)
    plt.title('Temperature Prediction: Actual vs Predicted')
    plt.xlabel('Time Steps')
    plt.ylabel('Outdoor Temperature')
    plt.legend()
    plt.grid(True)

    # 绘制误差
    plt.subplot(2, 1, 2)
    errors = y_test_original[start_idx:end_idx] - y_pred_original[start_idx:end_idx].flatten()
    plt.plot(errors, color='green', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Prediction Errors')
    plt.xlabel('Time Steps')
    plt.ylabel('Error')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('plots/prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Prediction results plot saved as 'plots/prediction_results.png'")

def setup_logging():
    """设置日志记录"""
    # 确保log文件夹存在
    os.makedirs('log', exist_ok=True)

    # 创建日志文件名（包含时间戳）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = "log/training_log_{}.txt".format(timestamp)

    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

    logger = logging.getLogger()
    logger.info("Training log started - {}".format(timestamp))
    logger.info("=" * 60)

    return logger, log_filename

def main():
    """
    主函数：执行完整的LSTM模型训练流程
    """
    # 设置日志
    logger, log_filename = setup_logging()

    logger.info("Starting LSTM Temperature Prediction Training")
    logger.info("=" * 60)

    # 1. 加载和预处理数据
    logger.info("Step 1: Loading and preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_preprocess_data()

    # 2. 构建模型
    logger.info("Step 2: Building LSTM model...")
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    model = build_lstm_model(input_shape)  # 使用默认参数：lstm_units=96, dropout_rate=0.25

    # 3. 训练模型
    logger.info("Step 3: Training model...")
    model, history = train_model(X_train, y_train, X_val, y_val, X_test, y_test, model,
                                epochs=100, batch_size=32)

    # 4. 评估模型
    logger.info("Step 4: Evaluating model...")
    y_pred, mae, rmse, r2 = evaluate_model(model, X_test, y_test, scaler)

    # 获取原始尺度的测试标签用于可视化
    y_test_original = y_test * scaler['y_scale'] + scaler['y_mean']

    # 5. 可视化结果
    logger.info("Step 5: Generating visualizations...")
    try:
        plot_training_history(history)
        plot_predictions(y_test_original, y_pred, num_samples=200)
        logger.info("Visualization plots saved")
    except ImportError:
        logger.warning("Matplotlib not available for plotting")
    except Exception as e:
        logger.error("Error creating plots: {}".format(e))

    # 6. 保存最终训练的模型和标准化参数
    logger.info("Step 6: Saving model and scaler parameters...")
    os.makedirs('save', exist_ok=True)
    model.save('save/model.keras')  # 使用新的Keras格式
    logger.info("Model saved as 'save/model.keras'")

    # 保存标准化参数（用于后续预测）
    np.savez('save/scaler_params.npz',
             feature_means=scaler['feature_means'],
             feature_stds=scaler['feature_stds'],
             y_mean=scaler['y_mean'],
             y_scale=scaler['y_scale'])
    logger.info("Scaler parameters saved as 'save/scaler_params.npz'")

    logger.info("")
    logger.info("=" * 60)
    logger.info("LSTM Training Completed!")
    logger.info("Log saved to: {}".format(log_filename))
    logger.info("Final Test Results:")
    logger.info("  MAE: {:.4f}".format(mae))
    logger.info("  RMSE: {:.4f}".format(rmse))
    logger.info("  R²: {:.4f}".format(r2))
    logger.info("=" * 60)

if __name__ == "__main__":
    # 设置TensorFlow日志级别
    tf.get_logger().setLevel('ERROR')

    # 设置随机种子保证可重复性
    np.random.seed(42)
    tf.random.set_seed(42)

    main()
