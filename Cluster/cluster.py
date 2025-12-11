import argparse
import json
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os


def cluster(features_path, filenames_path, out_path, method='kmeans', n_clusters=6, pca_dim=50):
    features = np.load(features_path)
    with open(filenames_path, 'r', encoding='utf-8') as f:
        fnames = json.load(f)

    # normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # optional PCA
    if pca_dim is not None and pca_dim > 0:
        pca = PCA(n_components=min(pca_dim, X.shape[1]))
        X = pca.fit_transform(X)

    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)

    labels = model.fit_predict(X)

    # save
    os.makedirs(out_path, exist_ok=True)
    mapping = {fn: int(lbl) for fn, lbl in zip(fnames, labels)}
    with open(os.path.join(out_path, 'pred_labels.json'), 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f'saved predicted labels to {out_path}/pred_labels.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='outputs/features.npy')
    parser.add_argument('--filenames', default='outputs/filenames.json')
    parser.add_argument('--out', default='outputs')
    parser.add_argument('--method', default='kmeans')
    parser.add_argument('--n-clusters', type=int, default=6)
    parser.add_argument('--pca-dim', type=int, default=50)
    args = parser.parse_args()
    cluster(args.features, args.filenames, args.out, args.method, args.n_clusters, args.pca_dim)
