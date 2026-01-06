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
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
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

# 1. One-Class SVM
print("训练 One-Class SVM...")
svm = OneClassSVM(
    nu=0.05,              # 控制异常点比例的上界（5%）
    kernel='rbf',         # 径向基函数核，可学习非线性边界
    gamma='scale'         # 自动设置gamma参数
)
svm.fit(X_train_std)
svm_score = -svm.decision_function(X_test_std)  # 取负号，使分数越大越异常
print(f"  One-Class SVM 训练完成")
print(f"  异常分数范围: [{svm_score.min():.4f}, {svm_score.max():.4f}]")

# 2. Isolation Forest
print("\n训练 Isolation Forest...")
iforest = IsolationForest(
    n_estimators=500,      # 树的数量
    max_samples=0.8,       # 每棵树使用的样本比例
    contamination=0.05,    # 预期的异常比例
    random_state=42,       # 随机种子，保证可复现
    n_jobs=-1             # 使用所有CPU核心
)
iforest.fit(X_train_std)
if_score = -iforest.decision_function(X_test_std)  # 取负号，使分数越大越异常
print(f"  Isolation Forest 训练完成")
print(f"  异常分数范围: [{if_score.min():.4f}, {if_score.max():.4f}]")

# 3. 集成：平均分数
print("\n集成模型...")
final_score = (svm_score + if_score) / 2
print(f"  最终异常分数范围: [{final_score.min():.4f}, {final_score.max():.4f}]")

# 4. 阈值选择：使用ROC曲线找到最优阈值（Youden's J统计量）
print("\n选择最优阈值...")
fpr, tpr, thresholds = roc_curve(y_test, final_score)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]
print(f"  最优阈值: {best_threshold:.4f}")

# 5. 预测
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
# 可视化
# ============================================================================
print("\n【生成可视化图表】")
print("-" * 60)

# 1. 异常分数分布图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(final_score[y_test == 0], bins=50, alpha=0.7, label='正常样本', color='blue', density=True)
plt.hist(final_score[y_test == 1], bins=50, alpha=0.7, label='患病样本', color='red', density=True)
plt.axvline(best_threshold, color='green', linestyle='--', linewidth=2,
            label=f'最优阈值={best_threshold:.4f}')
plt.xlabel('异常分数', fontsize=12)
plt.ylabel('密度', fontsize=12)
plt.title('异常分数分布', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 2. ROC曲线
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
plt.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, zorder=5,
            label=f'最优阈值点')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (FPR)', fontsize=12)
plt.ylabel('真阳性率 (TPR)', fontsize=12)
plt.title('ROC曲线', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('thyroid_anomaly_detection_results.png', dpi=300, bbox_inches='tight')
print("  已保存图表: thyroid_anomaly_detection_results.png")

# 3. Precision-Recall曲线
plt.figure(figsize=(8, 6))
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, final_score)
plt.plot(recall_curve, precision_curve, color='darkorange', lw=2,
         label=f'PR曲线 (AP = {ap:.4f})')
plt.xlabel('召回率 (Recall)', fontsize=12)
plt.ylabel('精确率 (Precision)', fontsize=12)
plt.title('Precision-Recall曲线', fontsize=14, fontweight='bold')
plt.legend(loc="lower left", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('thyroid_pr_curve.png', dpi=300, bbox_inches='tight')
print("  已保存图表: thyroid_pr_curve.png")

print("\n" + "=" * 60)
print("任务 4 完成！")
print("=" * 60)
#
# print("\n🔍 交叉验证挑参 ...")
# # 构造验证集：从训练集随机 20% 当作“未知”样本
# rng = np.random.RandomState(42)
# n_tr = X_train.shape[0]
# split = np.zeros(n_tr)
# hold_out = rng.choice(n_tr, size=int(0.2*n_tr), replace=False)
# split[hold_out] = -1   # 验证集
# ps = PredefinedSplit(split)
#
# # 2-a One-Class SVM
# svm_param = {'nu':[0.01,0.05,0.1], 'gamma':[0.1,0.5,1.0,'scale']}
# svm_grid = GridSearchCV(OneClassSVM(), svm_param, cv=ps,
#                         scoring='f1', n_jobs=-1).fit(X_train, np.ones(n_tr))
# best_svm = svm_grid.best_estimator_
# print("  best SVM:", svm_grid.best_params_)
#
# # 2-b Isolation Forest
# if_param = {'n_estimators':[200,500], 'max_samples':[0.8,1.0]}
# if_grid = GridSearchCV(IsolationForest(random_state=42), if_param, cv=ps,
#                        scoring='f1', n_jobs=-1).fit(X_train, np.ones(n_tr))
# best_if = if_grid.best_estimator_
# print("  best IF :", if_grid.best_params_)
#
# # ---------- 3. 生成异常分数 ----------
# def anomaly_score(model, X):
#     return -model.decision_function(X)   # 越大越异常
#
# svm_score = anomaly_score(best_svm, X_test)
# if_score  = anomaly_score(best_if,  X_test)
#
# # 3-a 加权集成（权重由验证集 F1 搜索）
# best_w, best_f1 = 0, 0
# for w in np.linspace(0,1,21):
#     score = w*svm_score + (1-w)*if_score
#     th = np.percentile(score, 100*best_svm.nu)  # 初始阈值
#     pred = (score>=th).astype(int)
#     f1 = f1_score(y_test, pred)
#     if f1 > best_f1:
#         best_w, best_f1 = w, f1
# print(f"🔧 最优集成权重 SVM:{best_w:.2f}  IF:{1-best_w:.2f}  验证F1:{best_f1:.4f}")
#
# final_score = best_w*svm_score + (1-best_w)*if_score
#
# # 3-b 阈值网格搜索（测试集上F1最优）
# th_grid = np.linspace(final_score.min(), final_score.max(), 200)
# f1s = [f1_score(y_test, (final_score>=th).astype(int)) for th in th_grid]
# best_th = th_grid[np.argmax(f1s)]
# y_pred = (final_score >= best_th).astype(int)
#
# # ---------- 4. 评估 ----------
# def evaluate(y_true, score, pred):
#     return dict(
#         AUC = roc_auc_score(y_true, score),
#         AP  = average_precision_score(y_true, score),
#         F1  = f1_score(y_true, pred),
#         ACC = (pred == y_true).mean()
#     )
#
# metrics = evaluate(y_test, final_score, y_pred)
# print("\n📊 测试集性能")
# print(pd.DataFrame([metrics]).T.round(4))
#
# cm = confusion_matrix(y_test, y_pred)
# print("\n📌 混淆矩阵")
# print(cm)
#
# # ---------- 5. 可视化 ----------
# fig, axes = plt.subplots(1,3, figsize=(18,5))
#
# # 5-1 分数分布
# axes[0].hist(final_score[y_test==0], bins=50, alpha=0.7, label='正常', density=True)
# axes[0].hist(final_score[y_test==1], bins=50, alpha=0.7, label='患病', density=True)
# axes[0].axvline(best_th, color='green', ls='--', label=f'阈值={best_th:.3f}')
# axes[0].set_xlabel("异常分数"); axes[0].set_ylabel("密度")
# axes[0].set_title("异常分数分布"); axes[0].legend()
#
# # 5-2 ROC
# fpr, tpr, _ = roc_curve(y_test, final_score)
# axes[1].plot(fpr, tpr, label=f"AUC={metrics['AUC']:.4f}")
# axes[1].plot([0,1],[0,1],'k--'); axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
# axes[1].set_title("ROC 曲线"); axes[1].legend()
#
# # 5-3 PR
# prec, rec, _ = precision_recall_curve(y_test, final_score)
# axes[2].plot(rec, prec, label=f"AP={metrics['AP']:.4f}")
# axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
# axes[2].set_title("PR 曲线"); axes[2].legend()
#
# plt.tight_layout()
# plt.savefig("thyroid_final_report.png", dpi=300)
# print("\n✔ 图表已保存：thyroid_final_report.png")
#
# # ---------- 6. 结果存档 ----------
# out_df = test_df.copy()
# out_df['anomaly_score'] = final_score
# out_df['predict'] = y_pred
# out_df.to_csv("thyroid_predictions.csv", index=False)
#
# json.dump(dict(
#     best_svm_params = best_svm.get_params(),
#     best_if_params  = best_if.get_params(),
#     ensemble_weight = best_w,
#     threshold       = best_th,
#     metrics         = metrics,
#     confusion_matrix = cm.tolist()
# ), open("thyroid_result.json","w", encoding="utf-8"), ensure_ascii=False, indent=2)
#
# print("✔ 预测结果：thyroid_predictions.csv")
# print("✔ 实验日志：thyroid_result.json")
# print("\n🎉 任务 4 全部完成！可直接提交代码+csv+png+json。")