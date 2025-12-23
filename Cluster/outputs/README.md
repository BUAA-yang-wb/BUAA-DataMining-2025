# 聚类结果说明

本目录包含图像聚类任务的最终输出结果。

## 1. 聚类可视化 (`cluster_vis.png`)

该图像展示了使用 t-SNE 算法将高维图像特征降维到 2D 平面后的分布情况。

- **左图 (Predicted Clusters)**：展示了模型预测的聚类结果。不同颜色代表不同的预测簇（Cluster 0 - Cluster 5）。
- **右图 (Ground Truth)**：展示了图像的真实类别标签。不同颜色代表真实的 6 个类别（cable, transistor, leather, pill, bottle, tile）。

**如何解读：**
- 对比左右两图，如果左图的颜色分布与右图高度一致，说明聚类效果较好。
- 如果左图中同一个簇（同色点）在右图中对应多种颜色，说明发生了**混淆**。
- 如果右图中同一个类（同色点）在左图中被分成了多个簇，说明发生了**过度分割**。

## 2. 评估指标 (`evaluation.json`)

该文件包含了量化评估聚类质量的三个核心指标：

```json
{
  "ARI": 0.xx,
  "NMI": 0.xx,
  "Purity": 0.xx
}
```

- **ARI (Adjusted Rand Index)**: 调整兰德系数。衡量预测聚类与真实标签的一致性。范围 [-1, 1]，越接近 1 越好。
- **NMI (Normalized Mutual Information)**: 归一化互信息。衡量两个标签分布的共享信息量。范围 [0, 1]，越接近 1 越好。
- **Purity**: 纯度。计算每个簇中占比最大的真实类别的比例均值。范围 [0, 1]，越接近 1 越好。

## 3. 中间文件 (`intermediates/`)

为了保持输出目录整洁，特征矩阵、文件名列表、预测标签等中间文件已被归档至 `intermediates/` 目录下：
- `features.npy`: 提取的图像特征矩阵。
- `filenames.json`: 对应的图像文件名。
- `pred_labels.json`: 聚类算法输出的预测标签。
- `pipeline_summary.json`: 运行摘要。
