import json
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter
import os


def purity_score(y_true, y_pred):
    labels = np.unique(y_pred)
    total = 0
    for l in labels:
        idx = np.where(y_pred == l)[0]
        true_labels = y_true[idx]
        most_common = Counter(true_labels).most_common(1)[0][1]
        total += most_common
    return total / len(y_true)


def evaluate(gt_path, pred_path, filenames_path):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)
    with open(pred_path, 'r', encoding='utf-8') as f:
        pred = json.load(f)
    with open(filenames_path, 'r', encoding='utf-8') as f:
        fnames = json.load(f)

    y_true = np.array([gt[fn] for fn in fnames])
    uniq = sorted(list(set(y_true)))
    label2int = {l: i for i, l in enumerate(uniq)}
    y_true_int = np.array([label2int[l] for l in y_true])

    y_pred_int = np.array([pred[fn] for fn in fnames])

    ari = adjusted_rand_score(y_true_int, y_pred_int)
    nmi = normalized_mutual_info_score(y_true_int, y_pred_int)
    purity = purity_score(y_true_int, y_pred_int)

    out = {
        'ARI': float(ari),
        'NMI': float(nmi),
        'Purity': float(purity),
    }
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(os.path.join(os.path.dirname(pred_path), 'evaluation.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print('Evaluation:', out)

    return out

def visualize_clusters(features_path, pred_path, gt_path=None, filenames_path=None, save_path=None, show=True):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    try:
        import seaborn as sns
        has_sns = True
    except ImportError:
        has_sns = False
        print("Warning: seaborn not found, using matplotlib for plotting.")
    
    # 加载特征
    features = np.load(features_path)

    
    # 加载预测标签
    with open(pred_path, 'r', encoding='utf-8') as f:
        pred = json.load(f)
    
    # 准备预测标签数组
    if filenames_path:
        with open(filenames_path, 'r', encoding='utf-8') as f:
            fnames = json.load(f)
        pred_labels = np.array([pred[fn] for fn in fnames])
    else:
        pred_labels = np.array(list(pred.values()))

    # 准备真实标签数组
    gt_labels = None
    if gt_path and filenames_path:
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt = json.load(f)
        gt_labels = np.array([gt[fn] for fn in fnames])

    print("正在进行降维可视化 (t-SNE)... 这可能需要几秒钟")
    # 先用 PCA 降维到 50 (如果特征维度很高) 以加速 t-SNE
    if features.shape[1] > 50:
        features = PCA(n_components=50).fit_transform(features)
    
    # t-SNE 降维到 2D
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(features)

    # 绘图
    if gt_labels is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 子图1: 预测聚类
        if has_sns:
            sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=pred_labels, palette='tab10', s=60, alpha=0.8, ax=axes[0], legend='full')
        else:
            scatter = axes[0].scatter(X_2d[:,0], X_2d[:,1], c=pred_labels, cmap='tab10', s=60, alpha=0.8)
            axes[0].legend(*scatter.legend_elements(), title="Clusters")
            
        axes[0].set_title('Predicted Clusters (t-SNE)')
        
        # 子图2: 真实标签
        if has_sns:
            sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=gt_labels, palette='tab10', s=60, alpha=0.8, ax=axes[1], legend='full')
        else:
            # Map string labels to ints for matplotlib
            unique_gt = np.unique(gt_labels)
            label_map = {l: i for i, l in enumerate(unique_gt)}
            gt_ints = np.array([label_map[l] for l in gt_labels])
            scatter = axes[1].scatter(X_2d[:,0], X_2d[:,1], c=gt_ints, cmap='tab10', s=60, alpha=0.8)
            # Create legend manually or just skip for fallback
            axes[1].legend(handles=scatter.legend_elements()[0], labels=list(unique_gt), title="Classes")
            
        axes[1].set_title('Ground Truth (t-SNE)')
        
    else:
        plt.figure(figsize=(10, 8))
        if has_sns:
            sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=pred_labels, palette='tab10', s=60, alpha=0.8, legend='full')
        else:
            scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=pred_labels, cmap='tab10', s=60, alpha=0.8)
            plt.legend(*scatter.legend_elements(), title="Clusters")
            
        plt.title('Predicted Clusters (t-SNE)')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Cluster visualization saved to {save_path}')
    
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', default='cluster_labels.json')
    parser.add_argument('--pred', default='outputs/pred_labels.json')
    parser.add_argument('--filenames', default='outputs/filenames.json')
    parser.add_argument('--features', default='outputs/features.npy')
    parser.add_argument('--visualize', action='store_true', help='可视化聚类分布')
    parser.add_argument('--vis-save', default='outputs/cluster_vis.png', help='聚类分布图保存路径')
    args = parser.parse_args()
    
    evaluate(args.gt, args.pred, args.filenames)
    
    # 如果指定了 visualize，或者默认情况下我们想生成图 (用户要求"只输出可视化结果"可能意味着总是生成)
    # 这里我们保持 --visualize 标志，但在 run_pipeline 中会调用它
    if args.visualize:
        visualize_clusters(args.features, args.pred, gt_path=args.gt, filenames_path=args.filenames, save_path=args.vis_save, show=True)

