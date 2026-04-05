#!/usr/bin/env python3
"""
处理 FakeNewsNet CSV 数据

将 FakeNewsNet 的 CSV 文件转换为项目统一格式

使用方法:
    python3 scripts/process_fakenewsnet_csv.py --output_dir data/raw/fakenewsnet
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List


def process_fakenewsnet_csv(output_dir: str):
    """处理 FakeNewsNet CSV 数据"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # FakeNewsNet 位置
    fakenewsnet_dir = Path("/home/admin/FakeNewsNet/dataset")
    
    if not fakenewsnet_dir.exists():
        print(f"❌ FakeNewsNet 目录不存在：{fakenewsnet_dir}")
        return False
    
    print("=" * 60)
    print("处理 FakeNewsNet CSV 数据")
    print("=" * 60)
    print(f"\n数据源：{fakenewsnet_dir}")
    
    # 处理的数据
    all_samples = []
    
    # 处理每个 CSV 文件
    csv_files = {
        "gossipcop_fake": {"source": "gossipcop", "label": "fake"},
        "gossipcop_real": {"source": "gossipcop", "label": "real"},
        "politifact_fake": {"source": "politifact", "label": "fake"},
        "politifact_real": {"source": "politifact", "label": "real"},
    }
    
    for csv_name, info in csv_files.items():
        csv_path = fakenewsnet_dir / f"{csv_name}.csv"
        
        if not csv_path.exists():
            print(f"⚠️  未找到：{csv_path}")
            continue
        
        print(f"\n处理 {csv_name}.csv...")
        
        # 读取 CSV
        df = pd.read_csv(csv_path)
        print(f"  ✓ 读取 {len(df)} 条记录")
        
        # 处理每条记录
        for idx, row in df.iterrows():
            sample = {
                "id": f"{info['source']}_{info['label']}_{idx}",
                "source": info["source"],
                "label": info["label"],
                "text": row.get("title", "") or row.get("text", ""),
                "url": row.get("url", ""),
                "has_image": False,  # CSV 版本没有图像
                "image_path": None,
            }
            
            # 只保留有文本的样本
            if sample["text"] and str(sample["text"]).strip():
                all_samples.append(sample)
    
    print(f"\n总计：{len(all_samples)} 条样本")
    
    # 划分训练集/测试集 (80/20)
    from sklearn.model_selection import train_test_split
    
    train_samples, test_samples = train_test_split(
        all_samples,
        test_size=0.2,
        random_state=42,
        stratify=[s["label"] for s in all_samples],
    )
    
    print(f"  训练集：{len(train_samples)} 条")
    print(f"  测试集：{len(test_samples)} 条")
    
    # 保存数据
    processed_data = {
        "train": train_samples,
        "test": test_samples,
        "metadata": {
            "dataset": "FakeNewsNet",
            "source": "CSV files (text only)",
            "total_samples": len(all_samples),
            "label_distribution": {},
            "sources": ["gossipcop", "politifact"],
        }
    }
    
    # 标签分布
    from collections import Counter
    all_labels = [s["label"] for s in all_samples]
    label_counts = dict(Counter(all_labels))
    processed_data["metadata"]["label_distribution"] = label_counts
    
    # 保存
    print(f"\n保存到：{output_dir}")
    
    for split in ["train", "test"]:
        split_file = output_dir / f"{split}.json"
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(processed_data[split], f, ensure_ascii=False, indent=2)
        print(f"  ✓ {split}: {len(processed_data[split])} 条 → {split_file}")
    
    # 保存元数据
    meta_file = output_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(processed_data["metadata"], f, ensure_ascii=False, indent=2)
    print(f"  ✓ 元数据 → {meta_file}")
    
    # 打印统计
    print(f"\n" + "=" * 60)
    print("数据集统计:")
    print(f"  来源：{processed_data['metadata']['sources']}")
    print(f"  标签分布：{label_counts}")
    print(f"  训练集：{len(train_samples)} 条")
    print(f"  测试集：{len(test_samples)} 条")
    print(f"  总计：{len(all_samples)} 条")
    print(f"  注意：CSV 版本只有文本，没有图像")
    print("=" * 60)
    
    print(f"\n✅ FakeNewsNet CSV 数据处理完成！")
    print(f"\n下一步:")
    print(f"  1. 预处理：python3 scripts/preprocess.py --dataset fakenewsnet --output_dir data/processed")
    print(f"  2. 训练：python3 scripts/train.py --config experiments/config_fakenewsnet.yaml")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="处理 FakeNewsNet CSV 数据")
    parser.add_argument("--output_dir", type=str, default="data/raw/fakenewsnet",
                       help="输出目录")
    
    args = parser.parse_args()
    
    success = process_fakenewsnet_csv(args.output_dir)
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
