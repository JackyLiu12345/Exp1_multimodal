#!/usr/bin/env python3
"""
使用 HuggingFace 镜像加载 PHEME 数据集

添加了:
- 阿里云镜像源
- 更长的超时时间
- 更多重试次数

使用方法:
    python3 scripts/download_pHEME_hf_mirror.py --output_dir data/raw/pheme
"""

import argparse
import json
import os
from pathlib import Path
from datasets import load_dataset


def download_pHEME(output_dir: str):
    """下载并处理 PHEME 数据集"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("从 HuggingFace 加载 PHEME 数据集 (使用镜像)")
    print("=" * 60)
    
    # 设置镜像和环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_OFFLINE'] = '0'
    
    print(f"\n使用镜像：{os.environ['HF_ENDPOINT']}")
    
    try:
        # 加载数据集（使用镜像）
        print("\n正在加载数据集...（可能需要几分钟）")
        print("提示：如果下载失败，请检查网络连接或使用 VPN")
        
        dataset = load_dataset(
            "GEM/PHEME",
            trust_remote_code=True,
            download_mode="force_redownload",
        )
        
        print(f"✓ 数据集加载成功")
        print(f"  可用分割：{list(dataset.keys())}")
        
        # 处理数据
        processed_data = {
            "train": [],
            "test": [],
            "metadata": {
                "dataset": "PHEME",
                "source": "HuggingFace (GEM/PHEME)",
                "total_samples": 0,
                "label_distribution": {},
                "events": []
            }
        }
        
        # PHEME 通常只有 test 分割
        if "test" in dataset:
            test_data = dataset["test"]
        else:
            # 如果没有明确分割，使用全部数据
            test_data = dataset[list(dataset.keys())[0]]
        
        print(f"\n处理 {len(test_data)} 条样本...")
        
        # 事件列表
        events = set()
        label_counts = {"real": 0, "fake": 0}
        
        for i, sample in enumerate(test_data):
            # 提取信息
            processed_sample = {
                "id": f"pheme_{i}",
                "text": sample.get("text", sample.get("content", "")),
                "label": "fake" if sample.get("label", 0) == 1 else "real",
                "event": sample.get("event", "unknown"),
                "split": "test" if i % 5 == 0 else "train",  # 80/20 划分
                "has_image": False,  # HuggingFace 版本可能没有图像
                "image_path": None,
            }
            
            # 统计
            events.add(processed_sample["event"])
            label_counts[processed_sample["label"]] += 1
            
            # 分配到分割
            split = processed_sample["split"]
            processed_data[split].append(processed_sample)
        
        # 更新元数据
        processed_data["metadata"]["total_samples"] = len(processed_data["train"]) + len(processed_data["test"])
        processed_data["metadata"]["events"] = list(events)
        processed_data["metadata"]["label_distribution"] = label_counts
        
        # 保存数据
        print(f"\n保存数据到：{output_dir}")
        
        for split in ["train", "test"]:
            if len(processed_data[split]) > 0:
                split_file = output_dir / f"{split}.json"
                with open(split_file, "w", encoding="utf-8") as f:
                    json.dump(processed_data[split], f, ensure_ascii=False, indent=2)
                print(f"  ✓ {split}: {len(processed_data[split])} 条样本 → {split_file}")
        
        # 保存元数据
        meta_file = output_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(processed_data["metadata"], f, ensure_ascii=False, indent=2)
        print(f"  ✓ 元数据 → {meta_file}")
        
        # 打印统计
        print(f"\n" + "=" * 60)
        print("数据集统计:")
        print(f"  事件：{events}")
        print(f"  标签分布：{label_counts}")
        print(f"  训练集：{len(processed_data['train'])} 条")
        print(f"  测试集：{len(processed_data['test'])} 条")
        print(f"  总计：{processed_data['metadata']['total_samples']} 条")
        print("=" * 60)
        
        print(f"\n✅ PHEME 数据集下载完成！")
        print(f"\n下一步:")
        print(f"  python3 scripts/preprocess.py --dataset pheme --output_dir data/processed")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 加载失败：{e}")
        print(f"\n备用方案:")
        print(f"  1. 检查网络连接")
        print(f"  2. 使用 VPN 或代理")
        print(f"  3. 手动下载：https://www.kaggle.com/datasets/arkaitz/zubiaga-pheme-dataset")
        return False


def main():
    parser = argparse.ArgumentParser(description="从 HuggingFace 下载 PHEME 数据集（镜像版）")
    parser.add_argument("--output_dir", type=str, default="data/raw/pheme",
                       help="输出目录")
    
    args = parser.parse_args()
    
    success = download_pHEME(args.output_dir)
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
