#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试训练好的模型在测试集上的性能
使用24个特征（20个气象特征 + 4个周期性时间特征）
"""

import numpy as np
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available, cannot load models")
    TENSORFLOW_AVAILABLE = False
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available, plotting will be disabled")
    MATPLOTLIB_AVAILABLE = False
    plt = None
import os
import warnings
warnings.filterwarnings('ignore')

def load_model_and_scaler():
    """加载模型和标准化参数"""
    print("Loading model and scaler...")

    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required but not available")

    # 检查文件是否存在
    model_path = 'save/model.keras'  
    scaler_path = 'save/scaler_params.npz'

    if not os.path.exists(model_path):
        raise FileNotFoundError("model file '{}' not found. Please run train.py first.".format(model_path))

    if not os.path.exists(scaler_path):
        raise FileNotFoundError("scaler parameters file '{}' not found. Please run train.py first.".format(scaler_path))

    # 加载模型
    model = load_model(model_path)

    print("model loaded successfully from {}".format(model_path))

    # 加载标准化参数
    scaler_params = np.load(scaler_path)
    print("Scaler parameters loaded successfully from {}".format(scaler_path))

    return model, scaler_params

def load_test_data():
    """加载测试数据"""
    print("Loading test data...")

    # 检查测试数据是否存在
    test_data_path = 'data/test_data.npz'
    if not os.path.exists(test_data_path):
        raise FileNotFoundError("Test data file not found: {}".format(test_data_path))

    # 加载测试数据
    test_data = np.load(test_data_path)
    X_test = test_data['X']
    y_test_raw = test_data['y']  # 原始未经标准化的数据

    print("Test data loaded successfully")
    print("Test samples: {}".format(X_test.shape[0]))
    print("Sequence length: {}".format(X_test.shape[1]))
    print("Features per timestep: {}".format(X_test.shape[2]))
    print("Raw y_test range: [{:.2f}, {:.2f}]".format(y_test_raw.min(), y_test_raw.max()))

    return X_test, y_test_raw

def evaluate_model_performance(model, X_test, y_test_raw, scaler_params):
    """评估模型在测试集上的性能"""
    print("\n=== Evaluating Model Performance ===")

    # 使用保存的标准化参数对测试数据进行标准化
    print("Applying standardization to test data...")
    n_samples, n_timesteps, n_features = X_test.shape

    # 对X_test进行标准化（特征级别）
    X_test_scaled = X_test.copy()
    for i in range(n_features):
        X_test_scaled[:, :, i] = (X_test[:, :, i] - scaler_params['feature_means'][i]) / (scaler_params['feature_stds'][i] + 1e-8)

    # 对y_test进行标准化
    y_test_scaled = (y_test_raw - scaler_params['y_mean']) / (scaler_params['y_scale'] + 1e-8)

    # 进行预测（在标准化空间）
    print("Making predictions with model...")
    y_pred_scaled = model.predict(X_test_scaled, verbose=1).flatten()

    # 反标准化预测结果和真实值到原始尺度
    y_pred_original = y_pred_scaled * scaler_params['y_scale'] + scaler_params['y_mean']
    y_test_original = y_test_scaled * scaler_params['y_scale'] + scaler_params['y_mean']

    print("\nData scale information:")
    print("  Raw y_test range: [{:.2f}, {:.2f}]".format(y_test_raw.min(), y_test_raw.max()))
    print("  Standardized y_test range: [{:.2f}, {:.2f}]".format(y_test_scaled.min(), y_test_scaled.max()))
    print("  Original scale y_test range: [{:.2f}, {:.2f}]".format(y_test_original.min(), y_test_original.max()))
    print("  y_mean: {:.2f}, y_scale: {:.2f}".format(scaler_params['y_mean'], scaler_params['y_scale']))

    # 计算评估指标
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred_original)

    print("\n=== Test Results (Original Scale) ===")
    print("Mean Absolute Error (MAE): {:.4f}".format(mae))
    print("Root Mean Squared Error (RMSE): {:.4f}".format(rmse))
    print("R-squared (R²): {:.4f}".format(r2))

    # 性能评估
    print("\n=== Model Performance Assessment ===")
    if mae < 5.0:
        print("✓ MAE: Excellent (< 5.0°C)")
    elif mae < 10.0:
        print("✓ MAE: Good (< 10.0°C)")
    else:
        print("⚠ MAE: Needs improvement (> 10.0°C)")

    if r2 > 0.7:
        print("✓ R²: Excellent (> 0.7)")
    elif r2 > 0.5:
        print("✓ R²: Good (> 0.5)")
    else:
        print("⚠ R²: Needs improvement (< 0.5)")

    # 统计预测误差分布
    errors = y_test_original - y_pred_original
    print("\n=== Error Statistics ===")
    print("Mean error: {:.4f}".format(np.mean(errors)))
    print("Median error: {:.4f}".format(np.median(errors)))
    print("Error std: {:.4f}".format(np.std(errors)))
    print("Max positive error: {:.4f}".format(np.max(errors)))
    print("Max negative error: {:.4f}".format(np.min(errors)))

    # 计算准确率区间
    accuracy_5deg = np.mean(np.abs(errors) <= 5.0) * 100
    accuracy_10deg = np.mean(np.abs(errors) <= 10.0) * 100

    print("\n=== Accuracy Analysis ===")
    print("Predictions within ±5: {:.1f}%".format(accuracy_5deg))
    print("Predictions within ±10: {:.1f}%".format(accuracy_10deg))

    # 生成预测结果可视化
    plot_predictions(y_test_original, y_pred_original, num_samples=200)

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_pred': y_pred_original,
        'y_test': y_test_original,
        'errors': errors
    }

def plot_predictions(y_test_original, y_pred_original, num_samples=200):
    """
    绘制预测结果对比（原始尺度）

    Args:
        y_test_original: 真实值（原始尺度）
        y_pred_original: 预测值（原始尺度）
        num_samples: 显示的样本数量
    """
    print("\n=== Plotting Predictions ===")

    try:
        plt.figure(figsize=(15, 8))

        # 选择要显示的样本范围
        start_idx = 0
        end_idx = min(num_samples, len(y_test_original))

        # 绘制预测vs实际
        plt.subplot(2, 1, 1)
        plt.plot(y_test_original[start_idx:end_idx], label='Actual OT', color='blue', alpha=0.7)
        plt.plot(y_pred_original[start_idx:end_idx], label='Predicted OT', color='red', alpha=0.7)
        plt.title('Temperature Prediction: Actual vs Predicted (Test Set)')
        plt.xlabel('Test Samples')
        plt.ylabel('Outdoor Temperature')
        plt.legend()
        plt.grid(True)

        # 绘制误差
        plt.subplot(2, 1, 2)
        errors = y_test_original[start_idx:end_idx] - y_pred_original[start_idx:end_idx].flatten()
        plt.plot(errors, color='green', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('Prediction Errors (Test Set)')
        plt.xlabel('Test Samples')
        plt.ylabel('Error')
        plt.grid(True)

        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/test_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Prediction results plot saved as 'plots/test_predictions.png'")

    except ImportError:
        print("Warning: matplotlib not available, skipping plot generation")
    except Exception as e:
        print("Error creating prediction plot: {}".format(e))

def save_test_results(results, filename='save/test_results.txt'):
    """保存测试结果到文件"""
    print("\n=== Saving Test Results ===")

    # 确保save文件夹存在
    os.makedirs('save', exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Temperature Prediction - Test Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Features: 24 (20 meteorological + 4 cyclical)\n\n")

        # 计算额外的性能指标
        from sklearn.metrics import mean_absolute_percentage_error, explained_variance_score, median_absolute_error, max_error

        mape = mean_absolute_percentage_error(results['y_test'], results['y_pred']) * 100  # 转换为百分比
        explained_var = explained_variance_score(results['y_test'], results['y_pred'])
        medae = median_absolute_error(results['y_test'], results['y_pred'])
        max_err = max_error(results['y_test'], results['y_pred'])

        f.write("Performance Metrics:\n")
        f.write("- MAE: {:.4f}\n".format(results['mae']))
        f.write("- RMSE: {:.4f}\n".format(results['rmse']))
        f.write("- R²: {:.4f}\n".format(results['r2']))
        f.write("- MAPE: {:.2f}%\n".format(mape))
        f.write("- Median AE: {:.4f}\n".format(medae))
        f.write("- Explained Variance: {:.4f}\n".format(explained_var))
        f.write("- Max Error: {:.4f}\n\n".format(max_err))

        f.write("Error Statistics:\n")
        f.write("- Mean error: {:.4f}\n".format(np.mean(results['errors'])))
        f.write("- Median error: {:.4f}\n".format(np.median(results['errors'])))
        f.write("- Error std: {:.4f}\n".format(np.std(results['errors'])))
        f.write("- Max positive error: {:.4f}\n".format(np.max(results['errors'])))
        f.write("- Max negative error: {:.4f}\n\n".format(np.min(results['errors'])))

        accuracy_5deg = np.mean(np.abs(results['errors']) <= 5.0) * 100
        accuracy_10deg = np.mean(np.abs(results['errors']) <= 10.0) * 100

        f.write("Accuracy Analysis:\n")
        f.write("- Predictions within ±5: {:.1f}%\n".format(accuracy_5deg))
        f.write("- Predictions within ±10: {:.1f}%\n".format(accuracy_10deg))

    print("Test results saved to '{}'".format(filename))

def main():
    """主函数：执行测试"""
    print("🧪 Starting Model Testing")
    print("=" * 50)

    try:
        # 1. 加载模型和标准化参数
        model, scaler_params = load_model_and_scaler()

        # 2. 加载测试数据
        X_test, y_test = load_test_data()

        # 3. 评估模型性能
        results = evaluate_model_performance(model, X_test, y_test, scaler_params)

        # 4. 保存测试结果
        save_test_results(results)

        print("\n" + "=" * 50)
        print(" Model Testing Completed!")
        print(" Final Results:")
        print("   MAE: {:.4f}".format(results['mae']))
        print("   RMSE: {:.4f}".format(results['rmse']))
        print("   R²: {:.4f}".format(results['r2']))
        print("=" * 50)

    except Exception as e:
        print("❌ Error during testing: {}".format(e))
        print("Please ensure you have run train.py first and all required files exist.")

if __name__ == "__main__":
    main()
