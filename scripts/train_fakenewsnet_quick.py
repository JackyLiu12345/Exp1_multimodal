#!/usr/bin/env python3
"""
FakeNewsNet CSV 快速训练脚本

使用纯文本分类器进行快速测试

使用方法:
    python3 scripts/train_fakenewsnet_quick.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.text_classifier import TextOnlyFakeNewsClassifier
from models.dataset_loader import MultimodalDataset, create_dataloader

# 配置
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = str(PROJECT_ROOT / "data" / "raw" / "fakenewsnet")
OUTPUT_DIR = str(PROJECT_ROOT / "results" / "fakenewsnet_quick")
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 512

print("=" * 60)
print("FakeNewsNet CSV 快速训练")
print("=" * 60)

# 创建输出目录
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

# 加载分词器
print(f"\n加载分词器：FacebookAI/roberta-base")
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

# 加载数据集
print(f"\n加载数据集：{DATA_DIR}")

train_dataset = MultimodalDataset(
    data_path=f"{DATA_DIR}/train.json",
    text_tokenizer=tokenizer,
    max_text_length=MAX_LENGTH,
    augment=False,
)

test_dataset = MultimodalDataset(
    data_path=f"{DATA_DIR}/test.json",
    text_tokenizer=tokenizer,
    max_text_length=MAX_LENGTH,
    augment=False,
)

print(f"  训练集：{len(train_dataset)} 条")
print(f"  测试集：{len(test_dataset)} 条")

# 创建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    collate_fn=train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else None,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
)

# 创建模型
print(f"\n创建模型...")
model = TextOnlyFakeNewsClassifier(
    model_name="FacebookAI/roberta-base",
    num_labels=2,
    moe_num_experts=3,
    moe_lora_r=8,
    moe_lora_alpha=16.0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

param_stats = model.get_num_params()
print(f"  参数量：{param_stats['total']:,} (可训练：{param_stats['trainable']:,})")
print(f"  设备：{device}")

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 训练循环
print(f"\n开始训练 ({NUM_EPOCHS} epochs)...")
print("=" * 60)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0
    
    from tqdm import tqdm
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for batch in pbar:
        input_ids = batch["text_input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_loss=True,
        )
        
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 计算准确率
        predictions = outputs["probs"].argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct/total:.4f}"
        })
    
    avg_loss = total_loss / num_batches
    train_acc = correct / total
    
    # 测试
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["text_input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_loss=True,
            )
            
            test_loss += outputs["loss"].item()
            predictions = outputs["probs"].argmax(dim=-1)
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)
    
    test_acc = test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} 完成:")
    print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Test Loss:  {avg_test_loss:.4f}, Test Acc:  {test_acc:.4f}")
    print("-" * 60)
    
    # 保存检查点
    checkpoint_path = output_path / f"checkpoint_epoch{epoch+1}.pt"
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_acc": train_acc,
        "test_acc": test_acc,
    }, checkpoint_path)
    print(f"  ✓ 保存检查点：{checkpoint_path}")

print("\n" + "=" * 60)
print("训练完成！")
print(f"最终测试准确率：{test_acc:.4f}")
print(f"结果保存在：{output_path}")
print("=" * 60)

# 保存训练日志
log_file = output_path / "training_log.json"
with open(log_file, "w") as f:
    json.dump({
        "final_test_acc": test_acc,
        "final_test_loss": avg_test_loss,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }, f, indent=2)

print(f"\n✓ 训练日志：{log_file}")
