#!/usr/bin/env python3
"""
多模态虚假新闻数据预处理脚本

功能:
- 文本清洗和标准化
- 图像加载和预处理
- 构建训练/验证/测试分割
- 保存为统一格式

使用方法:
    python scripts/preprocess.py --dataset twitter15 --output_dir data/processed
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import hashlib


class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self, language: str = "en"):
        self.language = language
        
    def clean_tweet(self, text: str) -> str:
        """清洗推文文本"""
        # 移除 URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 移除用户提及
        text = re.sub(r'@\w+', '', text)
        
        # 移除特殊字符和多余空格
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize(self, text: str) -> str:
        """文本标准化"""
        if self.language == "en":
            text = text.lower()
        return text.strip()
    
    def process(self, text: str) -> str:
        """完整的文本处理流程"""
        text = self.clean_tweet(text)
        text = self.normalize(text)
        return text


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def load_and_preprocess(self, image_path: str) -> Optional[Image.Image]:
        """加载并预处理图像"""
        try:
            img = Image.open(image_path)
            
            # 转换为 RGB (移除 alpha 通道)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 调整大小
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
            
            return img
        except Exception as e:
            print(f"⚠️  图像加载失败 {image_path}: {e}")
            return None
    
    def save_preprocessed(self, img: Image.Image, output_path: str, quality: int = 95):
        """保存预处理后的图像"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, quality=quality)
        return str(output_path)


class DatasetPreprocessor:
    """数据集预处理器"""
    
    def __init__(self, dataset_name: str, output_dir: str):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.text_processor = TextPreprocessor(language="en" if "twitter" in dataset_name else "zh")
        self.image_processor = ImagePreprocessor()
        
    def load_raw_data(self, raw_dir: Path) -> Dict:
        """加载原始数据"""
        data = {}
        
        for split in ["train", "dev", "test"]:
            split_file = raw_dir / f"{split}.json"
            if split_file.exists():
                with open(split_file, "r", encoding="utf-8") as f:
                    data[split] = json.load(f)
                print(f"✓ 加载 {split} 集：{len(data[split])} 条样本")
            else:
                print(f"⚠️  未找到 {split}.json")
                data[split] = []
        
        # 加载元数据
        meta_file = raw_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                data["metadata"] = json.load(f)
        
        return data
    
    def process_sample(self, sample: Dict, save_images: bool = True) -> Optional[Dict]:
        """处理单个样本"""
        processed = {
            "id": sample.get("id", ""),
            "text": self.text_processor.process(sample.get("text", "")),
            "label": sample.get("label", "unknown"),
            "has_image": False,
            "image_path": None,
        }
        
        # 处理图像
        if save_images and sample.get("image_path"):
            img = self.image_processor.load_and_preprocess(sample["image_path"])
            if img:
                # 生成新的图像路径
                img_hash = hashlib.md5(sample["id"].encode()).hexdigest()[:8]
                new_img_path = self.output_dir / "images" / f"{img_hash}.jpg"
                self.image_processor.save_preprocessed(img, str(new_img_path))
                processed["has_image"] = True
                processed["image_path"] = str(new_img_path)
        
        return processed
    
    def process_split(self, samples: List[Dict], split_name: str, save_images: bool = True) -> List[Dict]:
        """处理一个数据分割"""
        processed_samples = []
        
        for i, sample in enumerate(samples):
            processed = self.process_sample(sample, save_images)
            if processed and processed["text"]:  # 跳过空文本
                processed["split"] = split_name
                processed_samples.append(processed)
            
            if (i + 1) % 100 == 0:
                print(f"  处理进度：{i+1}/{len(samples)}")
        
        return processed_samples
    
    def save_processed(self, data: Dict):
        """保存处理后的数据"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存各分割
        for split in ["train", "dev", "test"]:
            if split in data and len(data[split]) > 0:
                split_file = self.output_dir / f"{split}.json"
                with open(split_file, "w", encoding="utf-8") as f:
                    json.dump(data[split], f, ensure_ascii=False, indent=2)
                print(f"✓ 保存 {split} 集：{len(data[split])} 条样本")
        
        # 保存元数据
        if "metadata" in data:
            meta_file = self.output_dir / "metadata.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(data["metadata"], f, ensure_ascii=False, indent=2)
        
        # 保存数据集统计信息
        stats = self.compute_statistics(data)
        stats_file = self.output_dir / "statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✓ 保存统计信息")
    
    def compute_statistics(self, data: Dict) -> Dict:
        """计算数据集统计信息"""
        stats = {
            "dataset": self.dataset_name,
            "splits": {},
            "label_distribution": {},
            "image_statistics": {
                "total_with_image": 0,
                "total_without_image": 0,
            },
            "text_statistics": {
                "avg_length": 0,
                "max_length": 0,
                "min_length": float('inf'),
            }
        }
        
        all_samples = []
        for split in ["train", "dev", "test"]:
            if split in data and len(data[split]) > 0:
                stats["splits"][split] = len(data[split])
                all_samples.extend(data[split])
        
        if not all_samples:
            return stats
        
        # 标签分布
        for sample in all_samples:
            label = sample.get("label", "unknown")
            stats["label_distribution"][label] = stats["label_distribution"].get(label, 0) + 1
            
            # 图像统计
            if sample.get("has_image", False):
                stats["image_statistics"]["total_with_image"] += 1
            else:
                stats["image_statistics"]["total_without_image"] += 1
            
            # 文本长度统计
            text_len = len(sample.get("text", ""))
            stats["text_statistics"]["avg_length"] += text_len
            stats["text_statistics"]["max_length"] = max(stats["text_statistics"]["max_length"], text_len)
            stats["text_statistics"]["min_length"] = min(stats["text_statistics"]["min_length"], text_len)
        
        # 计算平均长度
        stats["text_statistics"]["avg_length"] /= len(all_samples)
        if stats["text_statistics"]["min_length"] == float('inf'):
            stats["text_statistics"]["min_length"] = 0
        
        return stats
    
    def process(self, save_images: bool = True):
        """完整的预处理流程"""
        raw_dir = Path("data/raw") / self.dataset_name
        
        if not raw_dir.exists():
            print(f"❌ 原始数据目录不存在：{raw_dir}")
            print(f"   请先运行：python scripts/collect_data.py --dataset {self.dataset_name}")
            return
        
        print(f"=" * 60)
        print(f"开始预处理：{self.dataset_name}")
        print(f"=" * 60)
        
        # 加载原始数据
        raw_data = self.load_raw_data(raw_dir)
        
        # 处理各分割
        processed_data = {}
        for split in ["train", "dev", "test"]:
            if split in raw_data and len(raw_data[split]) > 0:
                print(f"\n处理 {split} 集...")
                processed_data[split] = self.process_split(raw_data[split], split, save_images)
        
        # 保留元数据
        if "metadata" in raw_data:
            processed_data["metadata"] = raw_data["metadata"]
        
        # 保存处理后的数据
        self.save_processed(processed_data)
        
        print(f"\n" + "=" * 60)
        print(f"预处理完成！")
        print(f"输出目录：{self.output_dir.absolute()}")
        print(f"=" * 60)


def main():
    parser = argparse.ArgumentParser(description="多模态虚假新闻数据预处理")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["twitter15", "twitter16", "weibo20", "pheme"],
                       help="数据集名称")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                       help="输出目录")
    parser.add_argument("--no_save_images", action="store_true",
                       help="不保存预处理后的图像")
    
    args = parser.parse_args()
    
    preprocessor = DatasetPreprocessor(
        dataset_name=args.dataset,
        output_dir=args.output_dir + "/" + args.dataset
    )
    
    preprocessor.process(save_images=not args.no_save_images)


if __name__ == "__main__":
    main()
