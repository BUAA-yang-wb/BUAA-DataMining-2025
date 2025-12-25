# Time Series Weather Prediction Project

## 任务描述 (Task Description)

基于德国某气象站半年内的气象数据，实现时间序列预测模型。根据过去2小时（12个时间点）的天气情况，预测下一时间点的室外温度（OT）变化。

**任务目标**：
- 输入：过去2小时的21维气象特征序列
- 输出：下一时间点的室外温度预测值
- 模型类型：时间序列预测（单步预测）

## 数据集描述 (Dataset Description)

### 基本信息
- **数据来源**：德国气象站观测数据
- **时间跨度**：2020年1月1日 00:10 - 2020年7月1日 00:00（约半年）
- **采样频率**：每10分钟记录一次
- **总观测点数**：26,200个
- **特征维度**：21个气象指标 + 1个时间戳

### 气象特征列表
1. **大气压力指标**：p (mbar), VPmax (mbar), VPact (mbar), VPdef (mbar)
2. **温度指标**：T (degC), Tpot (K), Tdew (degC), Tlog (degC), OT (目标变量)
3. **湿度指标**：rh (%), sh (g/kg), H2OC (mmol/mol)
4. **风指标**：wv (m/s), max. wv (m/s), wd (deg)
5. **降水指标**：rain (mm), raining (s)
6. **辐射指标**：SWDR (W/m²), PAR (µmol/m²s), max. PAR (µmol/m²s)
7. **空气密度**：rho (g/m³)

### 数据质量
- 无缺失值
- 时间序列连续
- 特征值在合理气象范围内

## 数据划分方式 (Data Splitting Method)

### 异常值检测和修复
- **检测方法**：IQR（四分位距）方法
- **判定标准**：超出 Q1-10*IQR 到 Q3+10*IQR 范围的值（k=10，保守策略）
- **修复策略**：前向填充（ffill）+ 后向填充（bfill）
- **处理列**：所有数值列（排除时间戳、rain (mm)、raining (s)）
- **目的**：清理异常值，保持数据连续性

### 周期性时间特征编码
- **日周期特征**：day_cos, day_sin（24小时周期）
- **年周期特征**：year_cos, year_sin（12个月周期）
- **编码方法**：使用正弦余弦函数保持周期连续性
- **优势**：23:00和00:00在数值上接近，避免时间编码突变

### 滑动窗口处理
- **窗口大小**：12个时间步（对应2小时的历史数据）
- **滑动步长**：1个时间步
- **样本生成**：从26,200个观测点生成26,188个训练样本
- **输入维度**：(12, 24) - 12时间步 × 24特征（含周期性时间特征，不含目标变量OT）
- **输出维度**：单值预测（室外温度OT）

### 训练/验证/测试集划分
- **划分原则**：按时间顺序划分，保持时序特性
- **训练集比例**：70%（约18,332个样本）
- **验证集比例**：15%（约3,928个样本）
- **测试集比例**：15%（约3,928个样本）
- **分割点**：训练集结束于样本索引18,331，验证集结束于样本索引22,259
- **重叠检查**：三个数据集样本完全不重合

### 数据文件结构
```
data/
├── weather.csv          # 原始数据
├── train_data.npz       # 训练数据 (NumPy格式)
├── val_data.npz         # 验证数据 (NumPy格式)
├── test_data.npz        # 测试数据 (NumPy格式)
├── train_data.csv       # 训练数据 (CSV格式，可视化用)
├── val_data.csv         # 验证数据 (CSV格式，可视化用)
└── test_data.csv        # 测试数据 (CSV格式，可视化用)
```

### 数据格式说明
- **NPZ格式**：用于模型训练，包含X（输入特征）和y（目标值）
- **CSV格式**：用于数据检查，特征已展开为二维格式
- **数据状态**：异常值已修复，原始尺度保持，训练时进行时间结构保持的标准化
- **异常值处理**：IQR方法检测，前向/后向填充修复
- **标准化方式**：特征级标准化（保持时间序列结构），目标变量单独标准化
- **数据集**：train_data用于训练，val_data用于验证，test_data用于最终测试

### 输出文件说明

#### save/ 文件夹
- **`model.keras`**：训练好的LSTM模型（Keras原生格式）
- **`scaler_params.npz`**：标准化参数（特征均值、标准差，目标变量均值、标准差）
- **`test_results.txt`**：LSTM模型测试结果报告（运行test.py后生成）

#### plots/ 文件夹
- **`training_history.png`**：训练历史可视化图表（如果matplotlib可用）
- **`prediction_results.png`**：训练集预测结果对比图表（如果matplotlib可用）
- **`test_predictions.png`**：测试集预测结果对比图表（如果matplotlib可用）

#### log/ 文件夹
- **`training_log_YYYYMMDD_HHMMSS.txt`**：详细的训练日志文件

### 模型架构

#### **增强型LSTM（默认）**：
- **类型**：双层LSTM网络（优化复杂度平衡）
- **结构**：LSTM(96) → LSTM(48) → Dense(64) → Dense(1)
- **输入处理**：连续特征标准化（保持时间序列结构）
- **正则化**：L2正则化(0.003) + Dropout(0.25) + BatchNormalization
- **特殊设计**：时间步级别的特征标准化，目标变量单独标准化

#### **训练配置**：
- **优化器**：Adam (lr=0.01, gradient clipping=1.0)
- **损失函数**：MSE（均方误差）
- **批大小**：32
- **最大epochs**：100（早停机制，patience=15）
- **学习率调度**：ReduceLROnPlateau (factor=0.5, patience=8, min_lr=1e-6)
- **评估指标**：MAE、RMSE、R²、MAPE、Median AE、Explained Variance、Max Error

### 脚本说明
- **`data_preprocessing.py`**：数据预处理脚本，生成训练/验证/测试数据集
- **`train.py`**：LSTM模型训练脚本，包含数据加载、标准化、训练和评估
- **`test.py`**：独立的模型测试脚本，用于评估训练好的LSTM模型在测试集上的性能

### 使用流程
1. 运行 `python data_preprocessing.py` 生成数据集
2. 运行 `python train.py` 训练LSTM模型
3. 运行 `python test.py` 测试LSTM模型性能

**预测时需要**：`save/model.keras` 和 `save/scaler_params.npz` 用于新数据预测
