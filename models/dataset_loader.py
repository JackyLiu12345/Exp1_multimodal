#!/usr/bin/env python3
"""
多模态虚假新闻数据集加载器

提供 PyTorch DataLoader，支持:
- 文本 + 图像多模态输入
- 数据增强
- 批量加载
- 不平衡采样

使用方法:
    from models.dataset_loader import MultimodalDataset, create_dataloader
    
    train_dataset = MultimodalDataset("data/processed/twitter15/train.json")
    train_loader = create_dataloader(train_dataset, batch_size=16, shuffle=True)
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Optional, Tuple, Callable
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path


class MultimodalDataset(Dataset):
    """多模态虚假新闻数据集"""
    
    def __init__(
        self,
        data_path: str,
        text_tokenizer=None,
        max_text_length: int = 512,
        image_transform: Optional[Callable] = None,
        augment: bool = False,
    ):
        """
        初始化数据集
        
        Args:
            data_path: JSON 文件路径
            text_tokenizer: 文本分词器 (如 transformers.AutoTokenizer)
            max_text_length: 最大文本长度
            image_transform: 图像变换
            augment: 是否使用数据增强
        """
        self.data_path = Path(data_path)
        self.max_text_length = max_text_length
        self.augment = augment
        
        # 加载数据
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        
        print(f"✓ 加载数据集：{len(self.samples)} 条样本")
        
        # 文本分词器
        self.text_tokenizer = text_tokenizer
        
        # 图像变换
        if image_transform is None:
            if augment:
                self.image_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225]),
                ])
            else:
                self.image_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225]),
                ])
        else:
            self.image_transform = image_transform
        
        # 标签映射
        self.label2id = {"real": 0, "fake": 1}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # 自动检测标签类型
        if len(self.samples) > 0:
            first_label = self.samples[0].get("label", "")
            if first_label not in self.label2id:
                # 重新构建标签映射
                unique_labels = sorted(set(s.get("label", "") for s in self.samples))
                self.label2id = {label: i for i, label in enumerate(unique_labels)}
                self.id2label = {i: label for label, i in self.label2id.items()}
                print(f"  检测到标签：{unique_labels}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # 文本处理
        text = sample.get("text", "")
        if self.text_tokenizer:
            text_inputs = self.text_tokenizer(
                text,
                truncation=True,
                max_length=self.max_text_length,
                padding="max_length",
                return_tensors="pt",
            )
            text_input_ids = text_inputs["input_ids"].squeeze(0)
            attention_mask = text_inputs["attention_mask"].squeeze(0)
        else:
            text_input_ids = text
            attention_mask = torch.ones(self.max_text_length)
        
        # 图像处理
        has_image = sample.get("has_image", False)
        image_path = sample.get("image_path")
        
        if has_image and image_path and Path(image_path).exists():
            try:
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.image_transform(image)
            except Exception as e:
                # 图像加载失败，使用零图像
                image_tensor = torch.zeros(3, 224, 224)
                has_image = False
        else:
            image_tensor = torch.zeros(3, 224, 224)
            has_image = False
        
        # 标签处理
        label_str = sample.get("label", "unknown")
        label_id = self.label2id.get(label_str, 0)
        label = torch.tensor(label_id, dtype=torch.long)
        
        return {
            "id": sample.get("id", str(idx)),
            "text_input_ids": text_input_ids,
            "attention_mask": attention_mask,
            "image": image_tensor,
            "has_image": has_image,
            "label": label,
            "label_str": label_str,
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重 (用于处理不平衡数据)"""
        from collections import Counter
        
        labels = [s.get("label", "") for s in self.samples]
        label_ids = [self.label2id.get(l, 0) for l in labels]
        
        counter = Counter(label_ids)
        total = len(label_ids)
        
        num_classes = len(self.label2id)
        weights = torch.zeros(num_classes)
        
        for class_id, count in counter.items():
            weights[class_id] = total / (num_classes * count)
        
        return weights
    
    def get_label_distribution(self) -> Dict:
        """获取标签分布"""
        from collections import Counter
        
        labels = [s.get("label", "") for s in self.samples]
        return dict(Counter(labels))


def create_dataloader(
    dataset: MultimodalDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    use_weighted_sampling: bool = False,
    **kwargs
) -> DataLoader:
    """
    创建 DataLoader
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 数据加载线程数
        use_weighted_sampling: 是否使用加权采样 (处理不平衡)
        **kwargs: 其他 DataLoader 参数
    
    Returns:
        DataLoader
    """
    if use_weighted_sampling and shuffle:
        # 计算类别权重
        class_weights = dataset.get_class_weights()
        sample_weights = [
            class_weights[sample["label"].item()] 
            for sample in dataset
        ]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        **kwargs
    )
    
    print(f"✓ 创建 DataLoader: batch_size={batch_size}, num_workers={num_workers}")
    
    return dataloader


def collate_fn(batch: List[Dict]) -> Dict:
    """
    批量整理函数
    
    Args:
        batch: 样本列表
    
    Returns:
        批量后的字典
    """
    result = {
        "ids": [item["id"] for item in batch],
        "text_input_ids": torch.stack([item["text_input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "image": torch.stack([item["image"] for item in batch]),
        "has_image": torch.tensor([item["has_image"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "label_str": [item["label_str"] for item in batch],
    }
    
    return result


def create_train_val_test_loaders(
    data_dir: str,
    text_tokenizer=None,
    batch_size: int = 16,
    max_text_length: int = 512,
    num_workers: int = 4,
    use_weighted_sampling: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练/验证/测试 DataLoader
    
    Args:
        data_dir: 数据目录 (包含 train.json, dev.json, test.json)
        text_tokenizer: 文本分词器
        batch_size: 批次大小
        max_text_length: 最大文本长度
        num_workers: 数据加载线程数
        use_weighted_sampling: 是否使用加权采样
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    
    # 创建数据集
    train_dataset = MultimodalDataset(
        str(data_dir / "train.json"),
        text_tokenizer=text_tokenizer,
        max_text_length=max_text_length,
        augment=True,
    )
    
    val_dataset = MultimodalDataset(
        str(data_dir / "dev.json"),
        text_tokenizer=text_tokenizer,
        max_text_length=max_text_length,
        augment=False,
    )
    
    test_dataset = MultimodalDataset(
        str(data_dir / "test.json"),
        text_tokenizer=text_tokenizer,
        max_text_length=max_text_length,
        augment=False,
    )
    
    # 创建 DataLoader
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        use_weighted_sampling=use_weighted_sampling,
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        use_weighted_sampling=False,
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        use_weighted_sampling=False,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试代码
    from transformers import AutoTokenizer
    
    print("测试 MultimodalDataset...")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    
    # 创建数据集
    dataset = MultimodalDataset(
        "data/processed/twitter15/train.json",
        text_tokenizer=tokenizer,
        max_text_length=512,
        augment=True,
    )
    
    # 获取一个样本
    sample = dataset[0]
    print(f"\n样本示例:")
    print(f"  ID: {sample['id']}")
    print(f"  文本长度：{sample['text_input_ids'].shape}")
    print(f"  图像形状：{sample['image'].shape}")
    print(f"  有图像：{sample['has_image']}")
    print(f"  标签：{sample['label']} ({sample['label_str']})")
    
    # 创建 DataLoader
    loader = create_dataloader(dataset, batch_size=4, shuffle=True)
    
    # 获取一个批次
    batch = next(iter(loader))
    print(f"\n批次示例:")
    print(f"  text_input_ids: {batch['text_input_ids'].shape}")
    print(f"  image: {batch['image'].shape}")
    print(f"  label: {batch['label'].shape}")
    
    print(f"\n✓ 测试通过!")
