"""
Visualization Tools: Anomaly Detection Results Visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import numpy as np
from typing import Dict, List, Tuple
import os


class AnomalyVisualizer:
    """Anomaly Detection Results Visualizer"""

    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Set matplotlib parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        sns.set_style("whitegrid")

    def plot_anomaly_scores(self, scores: np.ndarray, labels: np.ndarray,
                          title: str = "Anomaly Score Distribution", save_name: str = "anomaly_scores.png"):
        """Plot anomaly score distribution"""
        plt.figure(figsize=(14, 6))

        # Normal and anomaly sample score distributions
        plt.subplot(1, 2, 1)
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]

        plt.hist(normal_scores, alpha=0.7, label='Normal', bins=50, color='blue', density=True)
        plt.hist(anomaly_scores, alpha=0.7, label='Anomaly', bins=50, color='red', density=True)
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Score Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Box plot
        plt.subplot(1, 2, 2)
        data_to_plot = [normal_scores, anomaly_scores]
        plt.boxplot(data_to_plot, labels=['Normal', 'Anomaly'])
        plt.ylabel('Anomaly Score')
        plt.title('Score Box Plot')
        plt.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curve(self, labels: np.ndarray, scores: np.ndarray,
                      title: str = "ROC Curve", save_name: str = "roc_curve.png"):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc_score = np.trapz(tpr, fpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='.3f', linewidth=2, color='blue')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', alpha=0.7)

        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add optimal threshold point
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro',
                label='.3f')

        plt.legend()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()

        return auc_score, optimal_threshold

    def plot_precision_recall_curve(self, labels: np.ndarray, scores: np.ndarray,
                                   title: str = "Precision-Recall Curve",
                                   save_name: str = "precision_recall.png"):
        """Plot precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(labels, scores)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, color='green', label='PR Curve')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Calculate AP (Average Precision)
        ap_score = np.trapz(precision, recall)
        plt.text(0.7, 0.9, '.3f', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))

        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()

        return ap_score

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            title: str = "Confusion Matrix", save_name: str = "confusion_matrix.png"):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                   yticklabels=['Actual Normal', 'Actual Anomaly'])

        plt.title(title)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')

        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()

        return cm

    def plot_training_history(self, train_losses: List[float], val_metrics: List[Dict] = None,
                            title: str = "Training History", save_name: str = "training_history.png"):
        """Plot training history"""
        plt.figure(figsize=(12, 5))

        # Training loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss', linewidth=2, color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Learning rate (if available)
        plt.subplot(1, 2, 2)
        plt.plot(range(len(train_losses)), [1e-4] * len(train_losses), label='Learning Rate', linewidth=2, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()

    def create_summary_report(self, metrics: Dict, save_name: str = "evaluation_report.txt"):
        """Create evaluation summary report"""
        report_path = os.path.join(self.save_dir, save_name)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Multimodal Anomaly Detection Evaluation Report\n")
            f.write("=" * 50 + "\n\n")

            f.write("Model Configuration:\n")
            f.write(f"- Transformer Depth: {metrics.get('depth', 'N/A')}\n")
            f.write(f"- Embedding Dimension: {metrics.get('embed_dim', 'N/A')}\n")
            f.write(f"- Training Epochs: {metrics.get('epochs', 'N/A')}\n\n")

            f.write("Training Information:\n")
            f.write(f"- Final Training Loss: {metrics.get('total_train_loss', 'N/A')}\n\n")

            # Task2 评估结果
            if any(key.startswith('task2_') for key in metrics.keys()):
                f.write("Task2 (Image Anomaly Detection) Results:\n")
                def format_metric(task, metric_name):
                    key = f"{task}_{metric_name}"
                    value = metrics.get(key, 'N/A')
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        return f"{value:.4f}"
                    else:
                        return str(value)

                f.write(f"- AUC Score: {format_metric('task2', 'auc')}\n")
                f.write(f"- Precision: {format_metric('task2', 'precision')}\n")
                f.write(f"- Recall: {format_metric('task2', 'recall')}\n")
                f.write(f"- F1 Score: {format_metric('task2', 'f1')}\n")
                f.write(f"- Accuracy: {format_metric('task2', 'accuracy')}\n\n")

            # Task4 评估结果
            if any(key.startswith('task4_') for key in metrics.keys()):
                f.write("Task4 (Numerical Anomaly Detection) Results:\n")
                f.write(f"- AUC Score: {format_metric('task4', 'auc')}\n")
                f.write(f"- Precision: {format_metric('task4', 'precision')}\n")
                f.write(f"- Recall: {format_metric('task4', 'recall')}\n")
                f.write(f"- F1 Score: {format_metric('task4', 'f1')}\n")
                f.write(f"- Accuracy: {format_metric('task4', 'accuracy')}\n\n")

            # Task2 Confusion Matrix
            if any(key.startswith('task2_') for key in metrics.keys()):
                f.write("Task2 (Image) Confusion Matrix:\n")
                cm2 = metrics.get('task2_confusion_matrix', None)
                if cm2 is not None:
                    f.write(f"[[{cm2[0,0]}, {cm2[0,1]}]\n")
                    f.write(f" [{cm2[1,0]}, {cm2[1,1]}]]\n")
                    f.write("Format: [[TN, FP], [FN, TP]]\n")
                f.write(f"- Threshold: {metrics.get('task2_threshold', 'N/A')}\n\n")

            # Task4 Confusion Matrix
            if any(key.startswith('task4_') for key in metrics.keys()):
                f.write("Task4 (Numerical) Confusion Matrix:\n")
                cm4 = metrics.get('task4_confusion_matrix', None)
                if cm4 is not None:
                    f.write(f"[[{cm4[0,0]}, {cm4[0,1]}]\n")
                    f.write(f" [{cm4[1,0]}, {cm4[1,1]}]]\n")
                    f.write("Format: [[TN, FP], [FN, TP]]\n")
                f.write(f"- Threshold: {metrics.get('task4_threshold', 'N/A')}\n\n")

            f.write("Dataset Information:\n")
            f.write(f"- Image Training Samples: {metrics.get('image_train_samples', 'N/A')}\n")
            f.write(f"- Image Test Samples: {metrics.get('image_test_samples', 'N/A')}\n")
            f.write(f"- Numerical Training Samples: {metrics.get('numerical_train_samples', 'N/A')}\n")
            f.write(f"- Numerical Test Samples: {metrics.get('numerical_test_samples', 'N/A')}\n")
            f.write(f"- Total Training Samples: {metrics.get('total_train_samples', 'N/A')}\n")
            f.write(f"- Total Test Samples: {metrics.get('total_test_samples', 'N/A')}\n")

        print(f"Evaluation report saved to: {report_path}")


def plot_model_comparison(results_dict: Dict[str, Dict], save_dir: str = "results"):
    """Plot comparison of different models"""
    os.makedirs(save_dir, exist_ok=True)

    models = list(results_dict.keys())
    auc_scores = [results_dict[m]['auc'] for m in models]
    f1_scores = [results_dict[m]['f1'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    bars1 = plt.bar(x - width/2, auc_scores, width, label='AUC', alpha=0.8, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('AUC Score')
    plt.title('AUC Comparison Across Models')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加数值标签
    for bar, score in zip(bars1, auc_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                '.3f', ha='center', va='bottom')

    plt.subplot(1, 2, 2)
    bars2 = plt.bar(x + width/2, f1_scores, width, label='F1', alpha=0.8, color='lightcoral')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.title('F1 Comparison Across Models')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加数值标签
    for bar, score in zip(bars2, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                '.3f', ha='center', va='bottom')

    plt.suptitle('Multimodal Anomaly Detection Model Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 示例用法
    visualizer = AnomalyVisualizer()

    # 模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_anomalies = 100

    # 生成模拟的异常分数
    normal_scores = np.random.normal(0.2, 0.1, n_samples - n_anomalies)
    anomaly_scores = np.random.normal(0.8, 0.2, n_anomalies)
    scores = np.concatenate([normal_scores, anomaly_scores])

    # 生成标签
    labels = np.concatenate([np.zeros(n_samples - n_anomalies), np.ones(n_anomalies)])

    # 打乱数据
    indices = np.random.permutation(len(scores))
    scores = scores[indices]
    labels = labels[indices]

    # 绘制各种图表
    visualizer.plot_anomaly_scores(scores, labels)
    auc, threshold = visualizer.plot_roc_curve(labels, scores)
    ap = visualizer.plot_precision_recall_curve(labels, scores)

    # 基于阈值的预测
    predictions = (scores > threshold).astype(int)
    visualizer.plot_confusion_matrix(labels, predictions)

