import argparse
import os
import json
import numpy as np

def run_extract(dataset, out, batch_size, device):
    from extract_features import extract
    extract(dataset, out, batch_size=batch_size, device=device)


def run_cluster(features_path, filenames_path, out, method, n_clusters, pca_dim):
    from cluster import cluster as cluster_fn
    cluster_fn(features_path, filenames_path, out, method=method, n_clusters=n_clusters, pca_dim=pca_dim)


def run_evaluate(gt, pred, filenames):
    from evaluate import evaluate
    return evaluate(gt, pred, filenames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset')
    parser.add_argument('--out', default='outputs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--method', default='kmeans')
    parser.add_argument('--n-clusters', type=int, default=6)
    parser.add_argument('--pca-dim', type=int, default=50)
    parser.add_argument('--gt', default='cluster_labels.json')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print('1) Extracting features...')
    run_extract(args.dataset, args.out, args.batch_size, args.device)

    features_path = os.path.join(args.out, 'features.npy')
    filenames_path = os.path.join(args.out, 'filenames.json')
    pred_path = os.path.join(args.out, 'pred_labels.json')

    print('2) Running clustering...')
    run_cluster(features_path, filenames_path, args.out, args.method, args.n_clusters, args.pca_dim)

    print('3) Evaluating...')
    eval_res = run_evaluate(args.gt, pred_path, filenames_path)

    summary = {
        'features': features_path,
        'pred_labels': pred_path,
        'evaluation': eval_res,
    }
    with open(os.path.join(args.out, 'pipeline_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('Pipeline finished. Summary saved to', os.path.join(args.out, 'pipeline_summary.json'))

    # 自动生成聚类分布图
    try:
        from evaluate import visualize_clusters
        vis_path = os.path.join(args.out, 'cluster_vis.png')
        # 传递 gt 和 filenames 以支持对比图
        visualize_clusters(features_path, pred_path, gt_path=args.gt, filenames_path=filenames_path, save_path=vis_path, show=False)
        print(f"Visualization saved to {vis_path}")
    except Exception as e:
        print('Cluster visualization failed:', e)
        import traceback
        traceback.print_exc()

    # 整理输出文件 (将中间文件移动到 intermediates 文件夹，只保留结果图和评估指标)
    intermediates_dir = os.path.join(args.out, 'intermediates')
    os.makedirs(intermediates_dir, exist_ok=True)
    
    files_to_move = ['features.npy', 'filenames.json', 'pred_labels.json', 'pipeline_summary.json']
    for fname in files_to_move:
        src = os.path.join(args.out, fname)
        dst = os.path.join(intermediates_dir, fname)
        if os.path.exists(src):
            # 如果目标存在则先删除，防止 rename 失败
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
    
    print(f"\n[Info] Intermediate files have been moved to: {intermediates_dir}")
    print(f"[Result] Visualization: {vis_path}")
    print(f"[Result] Metrics: {os.path.join(args.out, 'evaluation.json')}")



if __name__ == '__main__':
    main()
