import argparse
import os
import json
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np
from tqdm import tqdm


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract(dataset_dir, out_dir, batch_size=32, device='cpu'):
    device = torch.device(device)
    # torchvision>=0.13 建议使用 weights 参数，避免 pretrained 的弃用告警
    try:
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    except Exception:
        # 兼容旧版本 torchvision（无 ResNet50_Weights 枚举时）
        model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()

    transform = build_transform()

    fnames = sorted([f for f in os.listdir(dataset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    features = []

    with torch.no_grad():
        for i in range(0, len(fnames), batch_size):
            batch_files = fnames[i:i+batch_size]
            imgs = []
            for fn in batch_files:
                path = os.path.join(dataset_dir, fn)
                img = Image.open(path).convert('RGB')
                imgs.append(transform(img))
            x = torch.stack(imgs).to(device)
            out = model(x)  # shape (B, 2048, 1, 1)
            out = out.reshape(out.shape[0], -1).cpu().numpy()
            features.append(out)

    features = np.vstack(features)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'features.npy'), features)
    with open(os.path.join(out_dir, 'filenames.json'), 'w', encoding='utf-8') as f:
        json.dump(fnames, f, ensure_ascii=False, indent=2)
    print(f'Saved features {features.shape} and {len(fnames)} filenames to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset', help='dataset dir')
    parser.add_argument('--out', default='outputs', help='output dir')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    extract(args.dataset, args.out, args.batch_size, args.device)
