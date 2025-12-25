"""
Time Series Prediction Data Preprocessing Script (Simplified Version)
Process weather.csv dataset with data checking, sliding window creation, and train/test split
No feature standardization, preserving original data scales
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def detect_and_fix_outliers_iqr(df, columns_to_process=None, exclude_columns=None):
    """
    使用IQR方法检测并修复异常值（前向填充）

    Args:
        df (pd.DataFrame): 输入数据框
        columns_to_process (list): 要处理的列，None表示处理所有数值列
        exclude_columns (list): 要排除的列

    Returns:
        pd.DataFrame: 修复后的数据框
    """
    print("\n" + "="*60)
    print("OUTLIER DETECTION AND FIXING (IQR Method)")
    print("="*60)

    if exclude_columns is None:
        exclude_columns = ['date', 'rain (mm)', 'raining (s)']

    # 确定要处理的列
    if columns_to_process is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        columns_to_process = [col for col in numeric_columns if col not in exclude_columns]

    print("\nColumns to process: {}".format(len(columns_to_process)))
    print("   Columns: {}".format(', '.join(columns_to_process)))
    print("   Excluded: {}".format(', '.join(exclude_columns)))
    print("   IQR multiplier: k=10 (conservative outlier detection)")

    df_cleaned = df.copy()
    total_outliers = 0
    columns_with_outliers = []

    print("\nProcessing each column...")
    print("-" * 80)

    for col in columns_to_process:
        # 计算IQR统计量
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1

        # 计算边界（使用k=10减少误杀）
        lower_bound = Q1 - 10 * IQR
        upper_bound = Q3 + 10 * IQR

        # 识别异常值
        outliers = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
        n_outliers = outliers.sum()
        outlier_percentage = (n_outliers / len(df_cleaned)) * 100

        # 获取异常值的统计信息
        outlier_values = df_cleaned.loc[outliers, col] if n_outliers > 0 else []

        print("\n📈 Column: {}".format(col))
        print("   Statistics: Q1={:.3f}, Q3={:.3f}, IQR={:.3f}".format(Q1, Q3, IQR))
        print("   Bounds: [{:.3f}, {:.3f}]".format(lower_bound, upper_bound))

        if n_outliers > 0:
            print("   ⚠️  Outliers: {} ({:.2f}%)".format(n_outliers, outlier_percentage))
            print("   Outlier range: [{:.3f}, {:.3f}]".format(
                outlier_values.min() if len(outlier_values) > 0 else 0,
                outlier_values.max() if len(outlier_values) > 0 else 0))

            # 记录修复前的统计
            original_mean = df_cleaned[col].mean()
            original_std = df_cleaned[col].std()

            # 正确的前向填充异常值：使用pandas的fillna方法
            # 先将异常值设为NaN，然后使用ffill和bfill
            df_cleaned.loc[outliers, col] = np.nan
            # 前向填充
            df_cleaned[col] = df_cleaned[col].fillna(method='ffill')
            # 对于开头仍为NaN的值，使用后向填充
            df_cleaned[col] = df_cleaned[col].fillna(method='bfill')

            # 记录修复后的统计
            fixed_mean = df_cleaned[col].mean()
            fixed_std = df_cleaned[col].std()

            print("   ✅ Fixed using forward/backward fill")
            print("   Stats change: Mean {:.3f} → {:.3f}, Std {:.3f} → {:.3f}".format(
                original_mean, fixed_mean, original_std, fixed_std))

            total_outliers += n_outliers
            columns_with_outliers.append(col)
        else:
            print("   ✅ No outliers detected")

    print("\n" + "="*60)
    print("OUTLIER PROCESSING SUMMARY")
    print("="*60)
    print("📊 Total outliers fixed: {}".format(total_outliers))
    print("📁 Columns with outliers: {} / {}".format(len(columns_with_outliers), len(columns_to_process)))
    if columns_with_outliers:
        print("   Affected columns: {}".format(', '.join(columns_with_outliers)))
    print("🔧 Fix method: Forward fill (ffill) + Backward fill (bfill)")
    print("💡 Note: Time series continuity preserved")

    return df_cleaned

def load_and_check_data(file_path):
    """
    加载数据并进行初步检查

    Args:
        file_path (str): 数据文件路径

    Returns:
        pd.DataFrame: 加载的数据框
    """
    print("=== Step 1: Data Loading and Checking ===")

    try:
        # 尝试多种编码方式
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print("Data loaded successfully, encoding: {}".format(encoding))
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise UnicodeDecodeError("Unable to read file with any supported encoding")

        print("Data shape: {}".format(df.shape))
        print("Columns: {}, Rows: {}".format(len(df.columns), len(df)))
    except Exception as e:
        print("Data loading failed: {}".format(e))
        return None

    print("\nData basic information:")
    print(df.head())

    # 检查缺失值
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values found:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values")

    # 时间戳处理
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print("Time range: {} to {}".format(df['date'].min(), df['date'].max()))

    return df

def create_cyclical_features(df):
    """
    创建周期性时间特征（在异常值处理之后调用）

    Args:
        df (pd.DataFrame): 输入数据框（已处理异常值）

    Returns:
        pd.DataFrame: 添加了周期性特征的数据框
    """
    print("\n=== Step 1.5: Creating Cyclical Time Features ===")

    # 创建周期性时间特征
    df['hour'] = df['date'].dt.hour
    df['month'] = df['date'].dt.month

    # 日周期特征 (24小时周期)
    df['day_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
    df['day_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))

    # 年周期特征 (12个月周期)
    df['year_cos'] = np.cos(df['month'] * (2 * np.pi / 12))
    df['year_sin'] = np.sin(df['month'] * (2 * np.pi / 12))

    print("Cyclical features created:")
    print("  - day_cos, day_sin: Daily cyclical features (24h cycle)")
    print("  - year_cos, year_sin: Yearly cyclical features (12 month cycle)")
    print("  - Note: hour and month are not used as input features")

    # 验证周期性特征的正确性
    print("\nCyclical features validation:")
    sample_hours = [0, 6, 12, 18, 23]
    print("Hour -> (day_cos, day_sin):")
    for h in sample_hours:
        cos_val = np.cos(h * (2 * np.pi / 24))
        sin_val = np.sin(h * (2 * np.pi / 24))
        print("  {} -> ({:.3f}, {:.3f})".format(h, cos_val, sin_val))

    # 检查23点和0点是否接近
    hour_23_cos = np.cos(23 * (2 * np.pi / 24))
    hour_23_sin = np.sin(23 * (2 * np.pi / 24))
    hour_0_cos = np.cos(0 * (2 * np.pi / 24))
    hour_0_sin = np.sin(0 * (2 * np.pi / 24))

    distance = np.sqrt((hour_23_cos - hour_0_cos)**2 + (hour_23_sin - hour_0_sin)**2)
    print("Distance between hour 23 and 0: {:.4f} (should be small for cyclical encoding)")

    return df

def create_sliding_windows(data, window_size=12, target_col='OT'):
    """
    创建滑动窗口样本

    Args:
        data (pd.DataFrame): 输入数据
        window_size (int): 窗口大小（时间步数）
        target_col (str): 目标列名

    Returns:
        tuple: (X, y) 输入特征和目标值
    """
    print("\n=== Step 3: Sliding Window Creation (window size={}) ===".format(window_size))

    # 排除时间戳、目标列和不应作为输入特征的列
    exclude_cols = ['date', target_col, 'hour', 'month', 'day_of_year']  # 不使用原始时间特征
    feature_cols = [col for col in data.columns if col not in exclude_cols]

    print("Number of feature columns: {}".format(len(feature_cols)))
    print("Feature columns: {}".format(', '.join(feature_cols)))
    print("Excluded columns: {}".format(', '.join(exclude_cols)))
    print("Target variable: {}".format(target_col))

    # 获取数值数据
    features = data[feature_cols].values
    targets = data[target_col].values

    # 创建滑动窗口
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size])  # 输入：过去window_size个时间步的特征
        y.append(targets[i+window_size])     # 输出：下一个时间步的目标值

    X = np.array(X)
    y = np.array(y)

    print("Sliding window processing completed:")
    print("   Input shape: {} (samples×time_steps×features)".format(X.shape))
    print("   Output shape: {} (samples×1)".format(y.shape))

    return X, y, feature_cols

def split_train_val_test(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    划分训练集、验证集和测试集

    Args:
        X (np.array): 输入特征
        y (np.array): 目标值
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例（测试集比例 = 1 - train_ratio - val_ratio）

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\n=== Step 4: Train/Val/Test Split ===")

    # 计算分割点
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    # 顺序划分（保持时间序列顺序）
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]

    test_ratio = 1 - train_ratio - val_ratio

    print("Dataset split completed:")
    print("   Training set: {} samples ({:.1f}%)".format(X_train.shape[0], train_ratio * 100))
    print("   Validation set: {} samples ({:.1f}%)".format(X_val.shape[0], val_ratio * 100))
    print("   Test set: {} samples ({:.1f}%)".format(X_test.shape[0], test_ratio * 100))

    return X_train, X_val, X_test, y_train, y_val, y_test

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, output_dir='data'):
    """
    保存处理后的数据

    Args:
        X_train, X_val, X_test, y_train, y_val, y_test: 划分后的数据
        feature_cols (list): 特征列名
        output_dir (str): 输出目录
    """
    print("\n=== Step 5: Saving Processed Data ===")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存数据
    np.savez(os.path.join(output_dir, 'train_data.npz'),
             X=X_train, y=y_train)
    np.savez(os.path.join(output_dir, 'val_data.npz'),
             X=X_val, y=y_val)
    np.savez(os.path.join(output_dir, 'test_data.npz'),
             X=X_test, y=y_test)

    print("Data saving completed:")
    print("   Training data: {}".format(os.path.join(output_dir, 'train_data.npz')))
    print("   Validation data: {}".format(os.path.join(output_dir, 'val_data.npz')))
    print("   Test data: {}".format(os.path.join(output_dir, 'test_data.npz')))

    # 保存为CSV格式（可选，便于查看）
    # 将3D数组转换为2D用于保存
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_val_2d = X_val.reshape(X_val.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)

    # 创建列名
    columns = []
    for t in range(X_train.shape[1]):  # time steps
        for f in feature_cols:  # features
            columns.append('{}_t{}'.format(f, t))

    train_df = pd.DataFrame(X_train_2d, columns=columns)
    train_df['target_OT'] = y_train
    train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)

    val_df = pd.DataFrame(X_val_2d, columns=columns)
    val_df['target_OT'] = y_val
    val_df.to_csv(os.path.join(output_dir, 'val_data.csv'), index=False)

    test_df = pd.DataFrame(X_test_2d, columns=columns)
    test_df['target_OT'] = y_test
    test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)

    print("   CSV format data also saved (optional)")

def main():
    """
    主函数：执行完整的数据预处理流程
    """
    print("Starting time series data preprocessing")
    print("=" * 50)

    # 文件路径
    data_file = 'data/weather.csv'

    # 检查文件是否存在
    if not os.path.exists(data_file):
        print("Data file not found: {}".format(data_file))
        return

    # 1. 加载和检查数据
    df = load_and_check_data(data_file)
    if df is None:
        return

    # 1.5. 异常值检测和修复
    df = detect_and_fix_outliers_iqr(df)

    # 1.6. 创建周期性时间特征
    df = create_cyclical_features(df)

    # 2. 创建滑动窗口（步骤3）
    X, y, feature_cols = create_sliding_windows(df, window_size=12, target_col='OT')

    # 3. 划分训练/验证/测试集（步骤4）
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, train_ratio=0.7, val_ratio=0.15)

    # 4. 保存处理后的数据（步骤5）
    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, output_dir='data')

    print("\n" + "=" * 50)
    print("Data preprocessing completed!")
    print("\nProcessing summary:")
    print("- Original data: 26,200 observations")
    print("- Outlier detection: IQR method with forward/backward fill")
    print("- Sliding window: 12 time steps")
    print("- Training set: {} samples (70%)".format(X_train.shape[0]))
    print("- Validation set: {} samples (15%)".format(X_val.shape[0]))
    print("- Test set: {} samples (15%)".format(X_test.shape[0]))
    print("- Input dimension: (12, 24) - 12 time steps × 24 features (including cyclical time features)")
    print("- Output dimension: Single value prediction (outdoor temperature OT)")
    print("- Data status: Outliers fixed, not standardized (original scale preserved)")

    print("\nUsage instructions:")
    print("1. Data saved in data/ folder")
    print("2. Use train_data.npz for training, val_data.npz for validation")
    print("3. Use test_data.npz for final evaluation")
    print("4. CSV format files available for data inspection and visualization")
    print("5. Note: Add feature standardization as needed")

if __name__ == "__main__":
    main()
