"""
Independent model testing script
Load trained multimodal anomaly detection model and perform test evaluation
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import os
import argparse
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from multimodal_anomaly_detector import MultiModalAnomalyTransformer, load_image_anomaly_data, load_thyroid_data
from visualization import AnomalyVisualizer


def load_model_checkpoint(checkpoint_path: str, device: str = 'auto'):
    """Load model checkpoint"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Loading model checkpoint: {checkpoint_path}")

    # Rebuild model from checkpoint
    config = checkpoint.get('config', None)

    # Use default model configuration
    model_config = {
        'img_size': 224,
        'patch_size': 16,
        'in_chans': 3,
        'embed_dim': 384,
        'depth': 6,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'num_numerical_features': 6
    }

    # If config is an object, try to extract model-related parameters from it
    if config is not None and hasattr(config, '__dict__'):
        # Extract possible model parameters from config object
        model_config.update({
            'embed_dim': getattr(config, 'embed_dim', 384),
            'depth': getattr(config, 'depth', 6),
            'num_heads': getattr(config, 'num_heads', 6),
            'mlp_ratio': getattr(config, 'mlp_ratio', 4.0),
            'qkv_bias': getattr(config, 'qkv_bias', True),
            'drop_rate': getattr(config, 'drop_rate', 0.0),
            'attn_drop_rate': getattr(config, 'attn_drop_rate', 0.0),
            'num_numerical_features': getattr(config, 'num_numerical_features', 6)
        })

    model = MultiModalAnomalyTransformer(
        img_size=model_config['img_size'],
        patch_size=model_config['patch_size'],
        in_chans=model_config['in_chans'],
        num_numerical_features=model_config['num_numerical_features'],
        embed_dim=model_config['embed_dim'],
        depth=model_config['depth'],
        num_heads=model_config['num_heads'],
        mlp_ratio=model_config['mlp_ratio'],
        qkv_bias=model_config['qkv_bias'],
        drop_rate=model_config['drop_rate'],
        attn_drop_rate=model_config['attn_drop_rate']
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("Model loaded successfully")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, device, checkpoint


def load_test_data(image_category: str = 'hazelnut', val_split: float = 0.2):
    """Load test data (excluding validation set used during training)"""
    print("Loading test data...")
    print(f"Validation split ratio: {val_split}")

    # Load image data
    _, image_test_paths, image_test_labels = load_image_anomaly_data(
        'Image_Anomaly_Detection', image_category, train_normal_only=True
    )

    # Load numerical data
    _, numerical_test_data, numerical_test_labels = load_thyroid_data('thyroid')

    print(f"Original image test samples: {len(image_test_paths)}")
    print(f"Original numerical test samples: {len(numerical_test_data)}")

    # Apply the same stratified splitting as in training to exclude validation set
    from sklearn.model_selection import train_test_split

    # Process image data: stratified split to exclude validation set
    if len(image_test_paths) > 1:
        # Separate normal and anomaly indices
        normal_indices = [i for i, label in enumerate(image_test_labels) if label == 0]
        anomaly_indices = [i for i, label in enumerate(image_test_labels) if label == 1]

        # Take the same proportion from each class for validation (as in training)
        val_normal_count = max(1, int(len(normal_indices) * val_split))
        val_anomaly_count = max(1, int(len(anomaly_indices) * val_split))

        val_indices = normal_indices[:val_normal_count] + anomaly_indices[:val_anomaly_count]
        test_indices = [i for i in range(len(image_test_paths)) if i not in val_indices]

        # Keep only test set (exclude validation set)
        image_test_paths = [image_test_paths[i] for i in test_indices]
        image_test_labels = [image_test_labels[i] for i in test_indices]

    # Process numerical data: stratified split to exclude validation set
    if len(numerical_test_data) > 1:
        # Separate normal and anomaly indices
        normal_indices = [i for i, label in enumerate(numerical_test_labels) if label == 0]
        anomaly_indices = [i for i, label in enumerate(numerical_test_labels) if label == 1]

        # Take the same proportion from each class for validation (as in training)
        val_normal_count = max(1, int(len(normal_indices) * val_split))
        val_anomaly_count = max(1, int(len(anomaly_indices) * val_split))

        val_indices = normal_indices[:val_normal_count] + anomaly_indices[:val_anomaly_count]
        test_indices = [i for i in range(len(numerical_test_data)) if i not in val_indices]

        # Keep only test set (exclude validation set)
        numerical_test_data = numerical_test_data[test_indices]
        numerical_test_labels = np.array(numerical_test_labels)[test_indices]

    print(f"After excluding validation set:")
    print(f"  Image test samples: {len(image_test_paths)}")
    print(f"  Numerical test samples: {len(numerical_test_data)}")

    return (image_test_paths, image_test_labels,
            numerical_test_data, numerical_test_labels)


def create_test_dataloaders(image_test_paths, image_test_labels,
                           numerical_test_data, numerical_test_labels,
                           batch_size: int = 16):
    """Create test data loaders"""
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from multimodal_anomaly_detector import MultiModalDataset

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Image test loader
    image_test_loader = None
    if image_test_paths:
        image_test_dataset = MultiModalDataset(
            image_paths=image_test_paths,
            labels=image_test_labels,
            transform=transform
        )
        image_test_loader = DataLoader(image_test_dataset, batch_size=batch_size, shuffle=False)

    # Numerical test loader
    numerical_test_loader = None
    if numerical_test_data is not None:
        numerical_test_dataset = MultiModalDataset(
            numerical_data=numerical_test_data,
            labels=numerical_test_labels
        )
        numerical_test_loader = DataLoader(numerical_test_dataset, batch_size=batch_size, shuffle=False)

    return image_test_loader, numerical_test_loader


def evaluate_model(model, test_loader, device, task_name: str, threshold_percentile: float = 95.0):
    """Evaluate model performance"""
    print(f"Evaluating {task_name}...")

    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # Prepare data
            image_data = batch.get('image')
            numerical_data = batch.get('numerical')
            labels = batch.get('label', torch.zeros(len(batch['image'] if image_data is not None else batch['numerical'])))

            if image_data is not None:
                image_data = image_data.to(device)
            if numerical_data is not None:
                numerical_data = numerical_data.to(device)

            # Forward propagation
            reconstruction_loss, anomaly_score, _, _ = model(image_data, numerical_data)

            # Combine anomaly scores
            final_score = reconstruction_loss + anomaly_score

            all_scores.extend(final_score.cpu().numpy())
            all_labels.extend(labels.numpy())

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    if len(scores) == 0:
        print(f"⚠️ {task_name} has no test data")
        return None

    # Calculate AUC
    try:
        auc_score = roc_auc_score(labels, scores)
        print(f"AUC Score: {auc_score:.4f}")
    except ValueError as e:
        print(f"AUC calculation failed: {e}")
        auc_score = float('nan')

    # Calculate threshold and predictions
    threshold = np.percentile(scores, threshold_percentile)
    predictions = (scores > threshold).astype(int)

    # Calculate classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)

    print("Classification metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    results = {
        'task_name': task_name,
        'auc': auc_score,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'threshold': threshold,
        'scores': scores,
        'labels': labels,
        'predictions': predictions
    }

    return results


def create_comparison_report(results_list: list, save_path: str = None):
    """Create comparison report"""
    if not results_list:
        return

    print("\n" + "="*60)
    print("🎯 Multimodal Anomaly Detection Model Test Report")
    print("="*60)

    for results in results_list:
        if results is None:
            continue

        task_name = results['task_name']
        print(f"\n📊 {task_name}")
        print("-" * 30)

        def format_metric(metric_name):
            value = results.get(metric_name, 'N/A')
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return f"{value:.4f}"
            else:
                return str(value)

        print(f"AUC Score: {format_metric('auc')}")
        print(f"Precision: {format_metric('precision')}")
        print(f"Recall: {format_metric('recall')}")
        print(f"F1 Score: {format_metric('f1')}")
        print(f"Accuracy: {format_metric('accuracy')}")

    # Save detailed report
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Multimodal Anomaly Detection Model Test Report\n")
            f.write("="*50 + "\n\n")

            for results in results_list:
                if results is None:
                    continue

                task_name = results['task_name']
                f.write(f"{task_name}\n")
                f.write("-" * 30 + "\n")

                def format_metric(metric_name):
                    value = results.get(metric_name, 'N/A')
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        return f"{value:.4f}"
                    else:
                        return str(value)

                f.write(f"AUC Score: {format_metric('auc')}\n")
                f.write(f"Precision: {format_metric('precision')}\n")
                f.write(f"Recall: {format_metric('recall')}\n")
                f.write(f"F1 Score: {format_metric('f1')}\n")
                f.write(f"Accuracy: {format_metric('accuracy')}\n\n")
                f.write("\n\n")

        print(f"\n💾 Detailed report saved: {save_path}")


def plot_test_results(results_list: list, save_dir: str = "test_results"):
    """Plot test results"""
    if not results_list:
        return

    os.makedirs(save_dir, exist_ok=True)

    # Set matplotlib parameters
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 12
    sns.set_style("whitegrid")

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for i, results in enumerate(results_list):
        if results is None:
            continue

        scores = results['scores']
        labels = results['labels']
        task_name = results['task_name']

        # Anomaly score distribution
        ax = axes[i // 2, i % 2]
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]

        ax.hist(normal_scores, alpha=0.7, label='Normal samples', bins=50, color='blue', density=True)
        ax.hist(anomaly_scores, alpha=0.7, label='Anomaly samples', bins=50, color='red', density=True)
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{task_name} - Anomaly Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add AUC information on the plot
        auc = results['auc']
        ax.text(0.7, 0.9, '.3f', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))

    plt.suptitle('Multimodal Anomaly Detection Test Results Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n📈 Visualization charts saved to: {save_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Multimodal anomaly detection model testing')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Model checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Test batch size')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Computation device')
    parser.add_argument('--image_category', type=str, default='hazelnut',
                       help='Image dataset category')
    parser.add_argument('--save_dir', type=str, default='test_results',
                       help='Results save directory')
    parser.add_argument('--no_plots', action='store_true',
                       help='Do not generate visualization charts')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio used during training (default: 0.2)')

    args = parser.parse_args()

    print("🧪 Multimodal Anomaly Detection Model Testing")
    print("=" * 40)

    try:
        # 1. Load model
        model, device, checkpoint = load_model_checkpoint(args.checkpoint, args.device)

        # 2. Load test data
        test_data = load_test_data(args.image_category, args.val_split)
        image_test_loader, numerical_test_loader = create_test_dataloaders(
            *test_data, batch_size=args.batch_size
        )

        # 3. Evaluate model
        results_list = []

        # Evaluate Task2 (Image Anomaly Detection)
        if image_test_loader:
            task2_results = evaluate_model(model, image_test_loader, device,
                                         "Task2 (Image Anomaly Detection)")
            results_list.append(task2_results)

        # Evaluate Task4 (Numerical Anomaly Detection)
        if numerical_test_loader:
            task4_results = evaluate_model(model, numerical_test_loader, device,
                                         "Task4 (Numerical Anomaly Detection)")
            results_list.append(task4_results)

        # 4. Create comparison report
        report_path = os.path.join(args.save_dir, 'test_report.txt')
        create_comparison_report(results_list, report_path)

        # 5. Generate visualization (optional)
        if not args.no_plots:
            plot_test_results(results_list, args.save_dir)

        print("\n🎉 Testing completed!")
        print(f"📁 Results saved to: {args.save_dir}")

    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
