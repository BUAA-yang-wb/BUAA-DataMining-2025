import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import config

def get_transforms():
    """
    获取图像预处理转换
    """
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
    ])

class AnomalyDataset(Dataset):
    def __init__(self, root_dir, category, mode='train', transform=None, label_file=None):
        """
        Args:
            root_dir (string): 数据集根目录
            category (string): 类别名称 ('hazelnut' 或 'zipper')
            mode (string): 'train' 或 'test'
            transform (callable, optional): 可选的图像转换
            label_file (string, optional): 标签文件路径 (仅测试模式需要)
        """
        self.root_dir = root_dir
        self.category = category
        self.mode = mode
        self.transform = transform
        self.image_paths = []
        self.labels = [] # 0: Good (正常), 1: Bad (异常)

        if mode == 'train':
            # 训练模式：只加载 good 样本
            good_dir = os.path.join(root_dir, category, 'train', 'good')
            if os.path.exists(good_dir):
                for img_name in os.listdir(good_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(good_dir, img_name))
                        self.labels.append(0) # 正常样本标签为 0
            else:
                print(f"Warning: Training directory not found: {good_dir}")

        elif mode == 'test':
            # 测试模式：加载 test 文件夹，并从 json 读取标签
            test_dir = os.path.join(root_dir, category, 'test')
            
            # 加载标签数据
            label_data = {}
            if label_file and os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    label_data = json.load(f)
            
            if os.path.exists(test_dir):
                for img_name in os.listdir(test_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(test_dir, img_name)
                        self.image_paths.append(full_path)
                        
                        # 构造 json 中的 key，例如 "hazelnut/test/001.png"
                        # 注意：这里假设 json key 的格式是 "category/test/filename"
                        key = f"{category}/test/{img_name}"
                        
                        if key in label_data:
                            label_str = label_data[key]['label']
                            # 'good' -> 0, 'bad' -> 1
                            self.labels.append(0 if label_str == 'good' else 1)
                        else:
                            # 如果标签缺失，默认标记为异常(1)或者打印警告
                            # print(f"Warning: Label not found for {key}")
                            self.labels.append(1) 
            else:
                print(f"Warning: Test directory not found: {test_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个全黑图像作为 fallback
            return torch.zeros(3, config.IMG_SIZE, config.IMG_SIZE), 0

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path
