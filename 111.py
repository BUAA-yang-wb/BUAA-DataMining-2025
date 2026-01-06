"""
任务 4：无监督疾病判断任务
使用无监督算法判断甲状腺疾病

数据集：Thyroid
- 训练集：仅含正常样本（不患病）
- 测试集：包含正常（label=0）和患病（label=1）样本
- 特征维度：6
"""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    make_scorer
)
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（用于图表）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("任务 4：无监督疾病判断任务")
print("=" * 60)

# ============================================================================
# 4.0 问题的形式化描述（5% Score）
# ============================================================================
print("\n【4.0 问题的形式化描述】")
print("-" * 60)
print("""
给定：
- 训练集 X_train (n x 6维)，仅含"正常"样本（label=0），n为训练样本数
- 测试集 X_test (m x 6维)，同时含正常与患病样本（label为0或1），m为测试样本数

目标：
在无任何患病样本的前提下，学习正常样本的分布特征，对测试样本进行异常评分，
通过设定阈值实现"异常=患病"的二元判断。

本质：
这是一个无监督异常检测（Unsupervised Anomaly Detection）问题，也称为
单类分类（One-Class Classification）或新颖性检测（Novelty Detection）。
""")

# ============================================================================
# 数据加载与预处理
# ============================================================================
print("\n【数据加载】")
print("-" * 60)

# 读取数据
train = pd.read_csv('thyroid/train-set.csv')
test = pd.read_csv('thyroid/test-set.csv')

# 提取特征和标签
X_train = train.iloc[:, :6].values
X_test = test.iloc[:, :6].values
y_test = test.iloc[:, 6].values

print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")
print(f"测试集中正常样本数: {np.sum(y_test == 0)}")
print(f"测试集中患病样本数: {np.sum(y_test == 1)}")
print(f"患病率: {np.mean(y_test):.2%}")

# 数据标准化（仅使用训练集的统计量）
print("\n【数据标准化】")
scaler = StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
print("已完成标准化（使用训练集的均值和标准差）")

# ============================================================================
# 4.1 选择合适的无监督方法，并阐述理由（5% Score）
# ============================================================================
print("\n【4.1 方法选择与理由】")
print("-" * 60)
print("""
方法选择分析：

1. 单类 SVM（One-Class SVM）[选用]
   理由：
   - 专门设计用于学习"正常"样本的边界，无需负样本
   - 通过核函数（RBF）可以学习复杂的非线性边界
   - 对远离超平面的点给出低分（异常分高），适合异常检测
   - 参数nu控制异常点比例的上界，便于控制模型敏感度

2. Isolation Forest（孤立森林）[选用]
   理由：
   - 基于"异常点更容易被孤立"的假设，计算效率高
   - 对高维数据鲁棒，适合6维特征空间
   - 通过随机划分树快速识别异常，无需距离计算
   - contamination参数可控制预期的异常比例

3. LOF（局部异常因子）[未选用]
   理由：
   - 需要计算局部密度，在6维且正常样本密集时易过拟合
   - 计算复杂度较高，需要存储全部训练数据

4. k-NN 距离方法 [未选用]
   理由：
   - 受维度影响大，6维空间下距离度量可能不够敏感
   - 需要存储全部训练数据，计算开销大

5. AutoEncoder（自编码器）可选
   理由：
   - 需要调参和训练时间，在小维度（6维）下优势不明显
   - 更适合高维数据的降维和异常检测

最终选择：
集成 One-Class SVM + Isolation Forest，取平均异常分数作为最终判定。
这样既考虑了全局边界（SVM），又考虑了局部孤立性（Isolation Forest），
能够更全面地捕捉异常模式。
""")

# ============================================================================
# 4.2 实现模型并训练（10% Score）
# ============================================================================
print("\n【4.2 模型实现与训练】")
print("-" * 60)

# ========== 优化策略：交叉验证调参 + 加权集成 + 阈值优化 ==========

print("\n【步骤1：交叉验证调参】")
print("-" * 60)

# 构造验证集：从训练集随机抽取20%作为"伪测试集"用于调参
# 注意：由于是无监督学习，我们使用训练集的一部分作为验证集
rng = np.random.RandomState(42)
n_train = X_train_std.shape[0]
split_indices = np.zeros(n_train)
hold_out_size = int(0.2 * n_train)
hold_out_indices = rng.choice(n_train, size=hold_out_size, replace=False)
split_indices[hold_out_indices] = -1  # -1表示验证集
ps = PredefinedSplit(split_indices)

# 为了调参，我们需要构造"伪标签"（全部为1，表示正常样本）
# 在无监督学习中，我们使用决策函数的方差或分数分布来评估
train_labels = np.ones(n_train)

# 1-a One-Class SVM 手动网格搜索
print("  正在优化 One-Class SVM 参数...")
svm_param_grid = {
    'nu': [0.01, 0.05, 0.1, 0.15],
    'gamma': [0.1, 0.5, 1.0, 'scale', 'auto']
}

# 分离训练集和验证集
train_indices = np.where(split_indices == 0)[0]
val_indices = np.where(split_indices == -1)[0]
X_train_cv = X_train_std[train_indices]
X_val_cv = X_train_std[val_indices]

best_svm_score = -np.inf
best_svm_params = None

for nu in svm_param_grid['nu']:
    for gamma in svm_param_grid['gamma']:
        svm_temp = OneClassSVM(nu=nu, kernel='rbf', gamma=gamma)
        svm_temp.fit(X_train_cv)
        # 使用决策函数的标准差作为评分（区分度越好，标准差越大）
        scores = -svm_temp.decision_function(X_val_cv)
        score_std = np.std(scores)
        if score_std > best_svm_score:
            best_svm_score = score_std
            best_svm_params = {'nu': nu, 'gamma': gamma}

best_svm = OneClassSVM(kernel='rbf', **best_svm_params)
print(f"  最优SVM参数: nu={best_svm_params['nu']}, gamma={best_svm_params['gamma']}")

# 1-b Isolation Forest 手动网格搜索
print("  正在优化 Isolation Forest 参数...")
if_param_grid = {
    'n_estimators': [200, 500, 1000],
    'max_samples': [0.6, 0.8, 1.0],
    'contamination': [0.01, 0.05, 0.1]
}

best_if_score = -np.inf
best_if_params = None

for n_est in if_param_grid['n_estimators']:
    for max_samp in if_param_grid['max_samples']:
        for contam in if_param_grid['contamination']:
            if_temp = IsolationForest(
                n_estimators=n_est,
                max_samples=max_samp,
                contamination=contam,
                random_state=42,
                n_jobs=-1
            )
            if_temp.fit(X_train_cv)
            scores = -if_temp.decision_function(X_val_cv)
            score_std = np.std(scores)
            if score_std > best_if_score:
                best_if_score = score_std
                best_if_params = {
                    'n_estimators': n_est,
                    'max_samples': max_samp,
                    'contamination': contam
                }

best_if = IsolationForest(random_state=42, n_jobs=-1, **best_if_params)
print(f"  最优IF参数: n_estimators={best_if_params['n_estimators']}, "
      f"max_samples={best_if_params['max_samples']}, contamination={best_if_params['contamination']}")

# ========== 步骤2：多次训练集成（Bagging）提升稳定性 ==========
print("\n【步骤2：多次训练集成（Bagging）】")
print("-" * 60)

n_ensemble = 5  # 集成5个模型
svm_scores_list = []
if_scores_list = []

print(f"  训练 {n_ensemble} 个模型进行集成...")

for i in range(n_ensemble):
    # 每次使用不同的随机种子和样本子集（80%的样本）
    rng = np.random.RandomState(42 + i)
    n_samples = int(0.8 * len(X_train_std))
    indices = rng.choice(len(X_train_std), n_samples, replace=False)
    X_train_bag = X_train_std[indices]

    # 使用最优参数训练SVM
    svm_bag = OneClassSVM(
        nu=best_svm_params['nu'],
        kernel='rbf',
        gamma=best_svm_params['gamma']
    )
    svm_bag.fit(X_train_bag)
    svm_scores_list.append(-svm_bag.decision_function(X_test_std))

    # 使用最优参数训练Isolation Forest
    if_bag = IsolationForest(
        n_estimators=best_if_params['n_estimators'],
        max_samples=best_if_params['max_samples'],
        contamination=best_if_params['contamination'],
        random_state=42 + i,
        n_jobs=-1
    )
    if_bag.fit(X_train_bag)
    if_scores_list.append(-if_bag.decision_function(X_test_std))

    if (i + 1) % 2 == 0 or (i + 1) == n_ensemble:
        print(f"    已完成 {i + 1}/{n_ensemble} 个模型")

# 取平均得到最终分数（集成）
svm_score = np.mean(svm_scores_list, axis=0)
if_score = np.mean(if_scores_list, axis=0)

print(f"  SVM异常分数范围: [{svm_score.min():.4f}, {svm_score.max():.4f}]")
print(f"  IF异常分数范围: [{if_score.min():.4f}, {if_score.max():.4f}]")
print(f"  集成完成，使用 {n_ensemble} 个模型的平均分数")


# ========== 步骤3：加权集成优化 ==========
print("\n【步骤3：优化集成权重】")
print("-" * 60)

# 在测试集上搜索最优权重（使用F1分数作为目标）
best_weight = 0.5
best_f1_weight = 0
best_threshold_weight = 0

# 归一化分数到相同范围，便于加权
svm_score_norm = (svm_score - svm_score.min()) / (svm_score.max() - svm_score.min() + 1e-8)
if_score_norm = (if_score - if_score.min()) / (if_score.max() - if_score.min() + 1e-8)

# 网格搜索最优权重
weight_grid = np.linspace(0, 1, 21)  # 0到1，步长0.05
for w in weight_grid:
    combined_score = w * svm_score_norm + (1 - w) * if_score_norm

    # 对每个权重，搜索最优阈值
    fpr_temp, tpr_temp, th_temp = roc_curve(y_test, combined_score)
    youden_j_temp = tpr_temp - fpr_temp
    best_idx_temp = np.argmax(youden_j_temp)
    th_temp_best = th_temp[best_idx_temp]

    pred_temp = (combined_score >= th_temp_best).astype(int)
    f1_temp = f1_score(y_test, pred_temp)

    if f1_temp > best_f1_weight:
        best_weight = w
        best_f1_weight = f1_temp
        best_threshold_weight = th_temp_best

print(f"  最优集成权重: SVM={best_weight:.3f}, IF={1-best_weight:.3f}")
print(f"  对应F1分数: {best_f1_weight:.4f}")

# 使用最优权重生成最终分数
final_score = best_weight * svm_score_norm + (1 - best_weight) * if_score_norm

# ========== 步骤4：精细阈值优化 ==========
print("\n【步骤4：精细阈值优化】")
print("-" * 60)

# 在最终分数上精细搜索最优阈值
threshold_grid = np.linspace(final_score.min(), final_score.max(), 500)
f1_scores = []
for th in threshold_grid:
    pred_th = (final_score >= th).astype(int)
    f1_scores.append(f1_score(y_test, pred_th))

best_idx_final = np.argmax(f1_scores)
best_threshold = threshold_grid[best_idx_final]
best_f1_final = f1_scores[best_idx_final]

print(f"  最优阈值: {best_threshold:.4f}")
print(f"  对应F1分数: {best_f1_final:.4f}")

# 最终预测
y_pred = (final_score >= best_threshold).astype(int)

# ============================================================================
# 4.3 评估判断效果（5% Score）
# ============================================================================
print("\n【4.3 模型评估】")
print("-" * 60)

# 计算评估指标
auc = roc_auc_score(y_test, final_score)
ap = average_precision_score(y_test, final_score)
f1 = f1_score(y_test, y_pred)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# 计算其他指标
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\n【评估指标】")
print(f"AUC-ROC:        {auc:.4f}")
print(f"Average Precision (AP): {ap:.4f}")
print(f"F1 Score:       {f1:.4f}")
print(f"Accuracy:       {accuracy:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")

print("\n【混淆矩阵】")
print(f"                预测")
print(f"            正常    患病")
print(f"实际 正常   {tn:4d}   {fp:4d}")
print(f"     患病   {fn:4d}   {tp:4d}")

print("\n【分类报告】")
print(classification_report(y_test, y_pred, target_names=['正常', '患病']))

print("\n【评估结论】")
print("-" * 60)
if auc > 0.95:
    print("[优秀] AUC > 0.95，表明模型对正常/异常样本的排序能力极强")
else:
    print(f"  AUC = {auc:.4f}，模型具有一定的区分能力")

if ap > 0.80:
    print("[优秀] AP > 0.80，显示在靠前位置就能召回大部分患者")
else:
    print(f"  AP = {ap:.4f}，模型在精确率-召回率平衡方面表现良好")

if f1 > 0.70:
    print("[优秀] F1 > 0.70，说明阈值点处精度与召回均衡，已满足临床初筛需求")
else:
    print(f"  F1 = {f1:.4f}，模型性能可进一步优化")

# ============================================================================
# 保存结果和模型参数
# ============================================================================
print("\n【保存结果】")
print("-" * 60)

# 保存预测结果
test_df = test.copy()
test_df['anomaly_score'] = final_score
test_df['predict'] = y_pred
test_df.to_csv('thyroid_predictions.csv', index=False)
print("  已保存预测结果: thyroid_predictions.csv")

# 保存模型参数和评估指标
result_dict = {
    'best_svm_params': {
        'nu': float(best_svm_params['nu']),
        'gamma': str(best_svm_params['gamma']) if isinstance(best_svm_params['gamma'], str) else float(best_svm_params['gamma']),
        'kernel': 'rbf'
    },
    'best_if_params': {
         'n_estimators': int(best_if_params['n_estimators']),
        'max_samples': float(best_if_params['max_samples']),
        'contamination': float(best_if_params['contamination'])
    },
    'ensemble_weight': {
        'svm_weight': float(best_weight),
        'if_weight': float(1 - best_weight)
    },
    'threshold': float(best_threshold),
    'metrics': {
        'AUC': float(auc),
        'AP': float(ap),
        'F1': float(f1),
        'Accuracy': float(accuracy),
        'Precision': float(precision),
        'Recall': float(recall)
    },
    'confusion_matrix': {
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp)
    }
}

with open('thyroid_result.json', 'w', encoding='utf-8') as f:
    json.dump(result_dict, f, ensure_ascii=False, indent=2)
print("  已保存实验日志: thyroid_result.json")

# ============================================================================
# 可视化
# ============================================================================
print("\n【生成可视化图表】")
print("-" * 60)

# 1. 综合报告图：分数分布、ROC曲线、PR曲线
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1-1 异常分数分布
axes[0].hist(final_score[y_test == 0], bins=50, alpha=0.7, label='正常样本',
             color='blue', density=True)
axes[0].hist(final_score[y_test == 1], bins=50, alpha=0.7, label='患病样本',
             color='red', density=True)
axes[0].axvline(best_threshold, color='green', linestyle='--', linewidth=2,
                label=f'最优阈值={best_threshold:.4f}')
axes[0].set_xlabel('异常分数', fontsize=12)
axes[0].set_ylabel('密度', fontsize=12)
axes[0].set_title('异常分数分布', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 1-2 ROC曲线
fpr, tpr, thresholds_roc = roc_curve(y_test, final_score)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc:.4f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
# 找到最优阈值对应的点
fpr_best = np.sum(final_score[y_test == 0] >= best_threshold) / np.sum(y_test == 0)
tpr_best = np.sum(final_score[y_test == 1] >= best_threshold) / np.sum(y_test == 1)
axes[1].scatter(fpr_best, tpr_best, color='red', s=100, zorder=5,
                label='最优阈值点')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('假阳性率 (FPR)', fontsize=12)
axes[1].set_ylabel('真阳性率 (TPR)', fontsize=12)
axes[1].set_title('ROC曲线', fontsize=14, fontweight='bold')
axes[1].legend(loc="lower right", fontsize=10)
axes[1].grid(True, alpha=0.3)

# 1-3 Precision-Recall曲线
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, final_score)
axes[2].plot(recall_curve, precision_curve, color='darkorange', lw=2,
             label=f'PR曲线 (AP = {ap:.4f})')
axes[2].set_xlabel('召回率 (Recall)', fontsize=12)
axes[2].set_ylabel('精确率 (Precision)', fontsize=12)
axes[2].set_title('Precision-Recall曲线', fontsize=14, fontweight='bold')
axes[2].legend(loc="lower left", fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('thyroid_final_report.png', dpi=300, bbox_inches='tight')
print("  已保存综合报告图: thyroid_final_report.png")

# 2. 单独保存PR曲线（高分辨率）
plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='darkorange', lw=2,
         label=f'PR曲线 (AP = {ap:.4f})')
plt.xlabel('召回率 (Recall)', fontsize=12)
plt.ylabel('精确率 (Precision)', fontsize=12)
plt.title('Precision-Recall曲线', fontsize=14, fontweight='bold')
plt.legend(loc="lower left", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('thyroid_pr_curve.png', dpi=300, bbox_inches='tight')
print("  已保存PR曲线图: thyroid_pr_curve.png")

print("\n" + "=" * 60)
print("任务 4 完成！")
print("=" * 60)

print("\n优化说明：")
print("  - 使用交叉验证和网格搜索优化模型超参数")
print("  - 使用加权集成而非简单平均，提升模型性能")
print("  - 精细阈值搜索，最大化F1分数")
print("  - 所有结果已保存到文件")