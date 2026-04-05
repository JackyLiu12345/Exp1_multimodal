#!/usr/bin/env python3
"""
多模态虚假新闻数据集收集脚本

支持的数据集:
- Twitter-15/16 (英文，文本 + 图片)
- Weibo-20 (中文，文本 + 图片)
- PHEME (英文，多事件)

使用方法:
    python scripts/collect_data.py --dataset twitter15 --output_dir data/raw
    python scripts/collect_data.py --dataset weibo20 --output_dir data/raw
    python scripts/collect_data.py --dataset pheme --output_dir data/raw

注意:
- Twitter-15/16 需要手动下载后解压到 data/downloads/
- Weibo-20 可从 GitHub 下载
- PHEME 需要从 Figshare 下载
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib


class Twitter1516Collector:
    """Twitter-15/16 数据集收集器"""
    
    def __init__(self, dataset: str = "twitter15", download_dir: str = "data/downloads"):
        assert dataset in ["twitter15", "twitter16"], "只支持 twitter15 或 twitter16"
        self.dataset = dataset
        self.download_dir = Path(download_dir)
        self.base_url = "https://www.cs.columbia.edu/~chenhao/resources/"
        
    def check_manual_download(self) -> bool:
        """检查用户是否已手动下载"""
        archive_name = f"{self.dataset}.tar.gz"
        archive_path = self.download_dir / archive_name
        return archive_path.exists()
    
    def provide_download_instructions(self):
        """提供手动下载说明"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Twitter-15/16 数据集下载说明                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  由于版权限制，需要手动下载数据集：                           ║
║                                                              ║
║  1. 访问下载页面：                                            ║
║     https://www.cs.columbia.edu/~chenhao/resources/          ║
║                                                              ║
║  2. 下载对应文件：                                            ║
║     - Twitter-15: {self.base_url}twitter15.tar.gz          ║
║     - Twitter-16: {self.base_url}twitter16.tar.gz          ║
║                                                              ║
║  3. 将下载的文件放到：                                        ║
║     {str(self.download_dir.absolute())}/{self.dataset}.tar.gz}                  ║
║                                                              ║
║  4. 重新运行此脚本                                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def extract_and_process(self, output_dir: Path) -> Dict:
        """解压并处理数据"""
        import tarfile
        
        archive_path = self.download_dir / f"{self.dataset}.tar.gz"
        
        # 创建临时解压目录
        temp_dir = self.download_dir / f"temp_{self.dataset}"
        temp_dir.mkdir(exist_ok=True)
        
        # 解压
        print(f"正在解压 {archive_path}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(temp_dir)
        
        # 处理数据
        processed_data = {
            "train": [],
            "dev": [],
            "test": [],
            "metadata": {
                "dataset": self.dataset,
                "total_samples": 0,
                "label_distribution": {}
            }
        }
        
        # 查找数据文件
        data_dir = temp_dir / self.dataset
        if not data_dir.exists():
            # 尝试其他目录结构
            for item in temp_dir.iterdir():
                if item.is_dir():
                    data_dir = item
                    break
        
        # 读取标签文件
        label_file = data_dir / f"{self.dataset}_label.txt"
        if not label_file.exists():
            # 尝试其他命名
            label_file = data_dir / "labels.txt"
        
        labels = {}
        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        tweet_id, label = parts[0], parts[1]
                        labels[tweet_id] = label
        
        # 读取文本和图像信息
        text_dir = data_dir / "text"
        image_dir = data_dir / "image"
        
        sample_count = 0
        for split in ["train", "dev", "test"]:
            split_file = data_dir / f"{split}.txt"
            if not split_file.exists():
                continue
                
            with open(split_file, "r", encoding="utf-8") as f:
                tweet_ids = [line.strip() for line in f if line.strip()]
            
            for tweet_id in tweet_ids:
                sample = {"id": tweet_id, "split": split}
                
                # 读取文本
                text_file = text_dir / f"{tweet_id}.txt"
                if text_file.exists():
                    with open(text_file, "r", encoding="utf-8") as f:
                        sample["text"] = f.read().strip()
                else:
                    sample["text"] = ""
                
                # 检查图像
                image_extensions = [".jpg", ".jpeg", ".png", ".gif"]
                sample["image_path"] = None
                for ext in image_extensions:
                    img_path = image_dir / f"{tweet_id}{ext}"
                    if img_path.exists():
                        sample["image_path"] = str(img_path)
                        break
                
                # 添加标签
                sample["label"] = labels.get(tweet_id, "unknown")
                
                processed_data[split].append(sample)
                sample_count += 1
        
        # 更新元数据
        processed_data["metadata"]["total_samples"] = sample_count
        
        # 统计标签分布
        all_labels = [s["label"] for s in processed_data["train"] + processed_data["dev"] + processed_data["test"]]
        label_counts = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        processed_data["metadata"]["label_distribution"] = label_counts
        
        # 清理临时文件
        shutil.rmtree(temp_dir)
        
        return processed_data


class Weibo20Collector:
    """Weibo-20 数据集收集器"""
    
    def __init__(self, download_dir: str = "data/downloads"):
        self.download_dir = Path(download_dir)
        self.github_url = "https://github.com/microsir/Weibo-20"
        
    def check_manual_download(self) -> bool:
        """检查是否已下载"""
        repo_dir = self.download_dir / "Weibo-20"
        return repo_dir.exists()
    
    def provide_download_instructions(self):
        """提供下载说明"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║              Weibo-20 数据集下载说明                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  从 GitHub 克隆数据集：                                        ║
║                                                              ║
║  1. 使用 git 克隆：                                           ║
║     git clone {self.github_url}.git                         ║
║                                                              ║
║  2. 移动到下载目录：                                          ║
║     mv Weibo-20 {str(self.download_dir.absolute())}/                          ║
║                                                              ║
║  或直接下载 ZIP:                                              ║
║  https://github.com/microsir/Weibo-20/archive/refs/heads/main.zip
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)


class PHEMECollector:
    """PHEME 数据集收集器"""
    
    def __init__(self, download_dir: str = "data/downloads"):
        self.download_dir = Path(download_dir)
        self.figshare_url = "https://figshare.com/articles/dataset/PHEME_dataset/6392939"
        
    def check_manual_download(self) -> bool:
        """检查是否已下载"""
        archive_path = self.download_dir / "PHEME.zip"
        data_dir = self.download_dir / "PHEME"
        return archive_path.exists() or data_dir.exists()
    
    def provide_download_instructions(self):
        """提供下载说明"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║              PHEME 数据集下载说明                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  PHEME 数据集需要从 Figshare 下载：                            ║
║                                                              ║
║  1. 访问：{self.figshare_url}                          ║
║                                                              ║
║  2. 点击 "Download" 按钮                                      ║
║                                                              ║
║  3. 下载后移动到：                                            ║
║     {str(self.download_dir.absolute())}/PHEME.zip                             ║
║                                                              ║
║  4. 重新运行此脚本，将自动解压和处理                           ║
║                                                              ║
║  数据集包含 5 个突发事件的谣言和非谣言推文 (共 9.7k 样本)        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def extract_and_process(self, output_dir: Path) -> Dict:
        """解压并处理数据"""
        import zipfile
        
        # 查找压缩包
        archive_path = self.download_dir / "PHEME.zip"
        if not archive_path.exists():
            # 检查是否已解压
            data_dir = self.download_dir / "PHEME"
            if data_dir.exists():
                print(f"✓ 检测到已解压的 PHEME 数据集")
                return self._process_pHEME_data(data_dir, output_dir)
            else:
                print(f"❌ 未找到 PHEME.zip 或 PHEME 目录")
                self.provide_download_instructions()
                return {"error": "Dataset not found"}
        
        # 解压
        print(f"正在解压 {archive_path}...")
        temp_dir = self.download_dir / "temp_pheme"
        temp_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # 查找数据目录
        data_dir = temp_dir / "PHEME"
        if not data_dir.exists():
            for item in temp_dir.iterdir():
                if item.is_dir():
                    data_dir = item
                    break
        
        return self._process_pHEME_data(data_dir, output_dir)
    
    def _process_pHEME_data(self, data_dir: Path, output_dir: Path) -> Dict:
        """处理 PHEME 数据"""
        processed_data = {
            "train": [],
            "test": [],
            "metadata": {
                "dataset": "PHEME",
                "total_samples": 0,
                "label_distribution": {},
                "events": []
            }
        }
        
        # PHEME 目录结构：每个事件一个文件夹
        events = []
        for item in data_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                events.append(item.name)
        
        processed_data["metadata"]["events"] = events
        print(f"检测到事件：{events}")
        
        sample_id = 0
        for event in events:
            event_dir = data_dir / event
            
            # 处理谣言和非谣言
            for label_type in ["rumours", "nonrumours"]:
                label_dir = event_dir / label_type
                if not label_dir.exists():
                    continue
                
                label = "fake" if label_type == "rumours" else "real"
                
                # 遍历每个样本
                for sample_dir in label_dir.iterdir():
                    if not sample_dir.is_dir():
                        continue
                    
                    sample = {
                        "id": f"{event}_{sample_id}",
                        "event": event,
                        "label": label,
                        "text": "",
                        "image_path": None,
                        "has_image": False,
                    }
                    
                    # 读取推文文本
                    reactions_dir = sample_dir / "reactions"
                    if reactions_dir.exists():
                        for reaction_file in reactions_dir.glob("*.json"):
                            try:
                                import json
                                with open(reaction_file, 'r', encoding='utf-8') as f:
                                    tweet_data = json.load(f)
                                    if "text" in tweet_data:
                                        sample["text"] = tweet_data["text"]
                                        break
                            except:
                                pass
                    
                    # 检查图像
                    if sample["text"]:  # 只有有文本的样本才保留
                        sample["split"] = "test"  # PHEME 通常用作测试集
                        processed_data["test"].append(sample)
                        sample_id += 1
        
        # 划分训练集 (使用其他事件)
        # 简单处理：将所有数据放入测试集
        processed_data["metadata"]["total_samples"] = len(processed_data["test"])
        
        # 标签分布
        from collections import Counter
        labels = [s["label"] for s in processed_data["test"]]
        processed_data["metadata"]["label_distribution"] = dict(Counter(labels))
        
        return processed_data


class FakeNewsNetCollector:
    """FakeNewsNet 数据集收集器"""
    
    def __init__(self, download_dir: str = "data/downloads"):
        self.download_dir = Path(download_dir)
        self.github_url = "https://github.com/KaiDMML/FakeNewsNet"
        self.repo_dir = self.download_dir / "FakeNewsNet"
        
    def check_cloned(self) -> bool:
        """检查是否已克隆"""
        return self.repo_dir.exists()
    
    def provide_clone_instructions(self):
        """提供克隆说明"""
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║           FakeNewsNet 数据集获取说明                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  FakeNewsNet 需要从 GitHub 克隆并使用 Twitter API 收集数据：   ║
║                                                              ║
║  1. 克隆仓库：                                                ║
║     cd {str(self.download_dir.absolute())}                                ║
║     git clone {self.github_url}.git                         ║
║                                                              ║
║  2. 安装依赖：                                                ║
║     cd FakeNewsNet                                            ║
║     pip install -r requirements.txt                           ║
║                                                              ║
║  3. 申请 Twitter API 密钥 (免费):                              ║
║     https://developer.twitter.com/                           ║
║                                                              ║
║  4. 配置 API 密钥：                                            ║
║     编辑 code/resources/tweet_keys_file.json                  ║
║                                                              ║
║  5. 收集数据：                                                ║
║     cd code                                                   ║
║     nohup python -m resource_server.app &> keys_server.out &  ║
║     nohup python main.py &> data_collection.out &             ║
║                                                              ║
║  6. 重新运行此脚本                                            ║
║                                                              ║
║  数据集包含 22k+ 样本 (GossipCop + PolitiFact)                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def process_fakenewsnet(self, output_dir: Path) -> Dict:
        """处理 FakeNewsNet 数据"""
        processed_data = {
            "train": [],
            "test": [],
            "metadata": {
                "dataset": "FakeNewsNet",
                "total_samples": 0,
                "label_distribution": {},
                "sources": ["gossipcop", "politifact"]
            }
        }
        
        # FakeNewsNet 目录结构
        # gossipcop/real/, gossipcop/fake/, politifact/real/, politifact/fake/
        
        sample_id = 0
        for source in ["gossipcop", "politifact"]:
            source_dir = self.repo_dir / "dataset" / source
            
            if not source_dir.exists():
                print(f"⚠️  未找到 {source} 目录，可能数据尚未收集")
                continue
            
            for label_type in ["real", "fake"]:
                label_dir = source_dir / label_type
                if not label_dir.exists():
                    continue
                
                label = "real" if label_type == "real" else "fake"
                
                # 遍历每个新闻样本
                for news_dir in label_dir.iterdir():
                    if not news_dir.is_dir():
                        continue
                    
                    sample = {
                        "id": f"{source}_{label_type}_{sample_id}",
                        "source": source,
                        "label": label,
                        "text": "",
                        "image_path": None,
                        "has_image": False,
                    }
                    
                    # 读取新闻内容
                    news_content_file = news_dir / "news_content.json"
                    if news_content_file.exists():
                        try:
                            import json
                            with open(news_content_file, 'r', encoding='utf-8') as f:
                                news_data = json.load(f)
                                sample["text"] = news_data.get("text", "")
                                sample["title"] = news_data.get("title", "")
                                sample["url"] = news_data.get("url", "")
                                
                                # 图像
                                images = news_data.get("images", [])
                                if images:
                                    # 保存第一张图像
                                    sample["image_urls"] = images
                        except Exception as e:
                            print(f"⚠️  读取 {news_content_file} 失败：{e}")
                    
                    # 检查是否有推文
                    tweets_dir = news_dir / "tweets"
                    if tweets_dir.exists():
                        tweet_files = list(tweets_dir.glob("*.json"))
                        sample["tweet_count"] = len(tweet_files)
                    
                    if sample["text"]:  # 只保留有文本的样本
                        # 简单划分：80% 训练，20% 测试
                        split = "train" if sample_id % 5 != 0 else "test"
                        sample["split"] = split
                        processed_data[split].append(sample)
                        sample_id += 1
        
        processed_data["metadata"]["total_samples"] = len(processed_data["train"]) + len(processed_data["test"])
        
        # 标签分布
        from collections import Counter
        all_labels = [s["label"] for s in processed_data["train"] + processed_data["test"]]
        processed_data["metadata"]["label_distribution"] = dict(Counter(all_labels))
        
        return processed_data


def save_processed_data(data: Dict, output_dir: Path, dataset_name: str):
    """保存处理后的数据"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为 JSON 文件
    for split in ["train", "dev", "test"]:
        if split in data and len(data[split]) > 0:
            split_file = output_dir / f"{split}.json"
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump(data[split], f, ensure_ascii=False, indent=2)
            print(f"✓ 保存 {split} 集：{len(data[split])} 条样本 → {split_file}")
    
    # 保存元数据
    meta_file = output_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(data["metadata"], f, ensure_ascii=False, indent=2)
    print(f"✓ 保存元数据 → {meta_file}")
    
    # 保存数据集信息
    info_file = output_dir / "dataset_info.json"
    dataset_info = {
        "name": dataset_name,
        "created_at": str(Path(output_dir).stat().st_mtime),
        "splits": [s for s in ["train", "dev", "test"] if s in data and len(data[s]) > 0],
        "total_samples": data["metadata"]["total_samples"],
        "label_distribution": data["metadata"]["label_distribution"],
    }
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="多模态虚假新闻数据集收集")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=["twitter15", "twitter16", "weibo20", "pheme", "fakenewsnet"],
                       help="数据集名称")
    parser.add_argument("--output_dir", type=str, default="data/raw",
                       help="输出目录")
    parser.add_argument("--download_dir", type=str, default="data/downloads",
                       help="下载缓存目录")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) / args.dataset
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 60)
    print(f"数据集收集：{args.dataset.upper()}")
    print(f"=" * 60)
    
    if args.dataset in ["twitter15", "twitter16"]:
        collector = Twitter1516Collector(args.dataset, str(download_dir))
        
        if not collector.check_manual_download():
            collector.provide_download_instructions()
            return
        else:
            print(f"✓ 检测到已下载的数据集")
            processed_data = collector.extract_and_process(output_dir)
            save_processed_data(processed_data, output_dir, args.dataset)
            
    elif args.dataset == "weibo20":
        collector = Weibo20Collector(str(download_dir))
        
        if not collector.check_manual_download():
            collector.provide_download_instructions()
            return
        else:
            print(f"✓ 检测到已下载的数据集，需要进一步处理...")
            # TODO: 实现 Weibo-20 的处理逻辑
            
    elif args.dataset == "pheme":
        collector = PHEMECollector(str(download_dir))
        
        if not collector.check_manual_download():
            collector.provide_download_instructions()
            return
        else:
            print(f"✓ 检测到已下载的 PHEME 数据集")
            processed_data = collector.extract_and_process(output_dir)
            if "error" not in processed_data:
                save_processed_data(processed_data, output_dir, args.dataset)
            
    elif args.dataset == "fakenewsnet":
        collector = FakeNewsNetCollector(str(download_dir))
        
        if not collector.check_cloned():
            collector.provide_clone_instructions()
            return
        else:
            print(f"✓ 检测到已克隆的 FakeNewsNet 仓库")
            processed_data = collector.process_fakenewsnet(output_dir)
            save_processed_data(processed_data, output_dir, args.dataset)
    
    print(f"\n" + "=" * 60)
    print(f"数据收集完成！")
    print(f"输出目录：{output_dir.absolute()}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
