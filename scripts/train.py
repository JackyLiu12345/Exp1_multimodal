#!/usr/bin/env python3
"""
多模态虚假新闻分类器训练脚本

支持:
- 多 GPU 训练
- 混合精度训练
- WandB/TensorBoard 日志
- 检查点保存
- 早停

使用方法:
    python scripts/train.py --config experiments/config.yaml
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from models.classifier import MultimodalFakeNewsClassifier
from models.dataset_loader import create_train_val_test_loaders
from models.moe_lora import MoELoRA

from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb 未安装，将禁用 WandB 日志")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        config: Dict,
        output_dir: str = "results",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.output_dir = Path(output_dir)
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备：{self.device}")
        
        # 混合精度
        self.use_fp16 = config.get("training", {}).get("fp16", True)
        self.scaler = GradScaler() if self.use_fp16 else None
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 日志
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        if TENSORBOARD_AVAILABLE and config.get("logging", {}).get("tensorboard", {}).get("enabled", True):
            tb_dir = self.log_dir / "tensorboard"
            self.writer = SummaryWriter(str(tb_dir))
        else:
            self.writer = None
        
        # WandB
        if WANDB_AVAILABLE and config.get("logging", {}).get("wandb", {}).get("enabled", True):
            wandb_config = config.get("logging", {}).get("wandb", {})
            self.wandb_run = wandb.init(
                project=wandb_config.get("project", "multimodal-fake-news"),
                entity=wandb_config.get("entity", ""),
                config=config,
                name=f"{config.get('dataset', {}).get('name', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
        else:
            self.wandb_run = None
        
        # 最佳模型跟踪
        self.best_metric = 0.0
        self.best_metric_name = config.get("evaluation", {}).get("monitor_metric", "f1")
        self.monitor_mode = config.get("evaluation", {}).get("monitor_mode", "max")
        
        # 早停
        self.early_stop_patience = config.get("training", {}).get("early_stop_patience", 3)
        self.early_stop_counter = 0
        
        # 训练统计
        self.global_step = 0
        self.epoch = 0
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        opt_config = self.config.get("training", {}).get("optimizer", {})
        
        lr = opt_config.get("lr", 2e-5)
        weight_decay = opt_config.get("weight_decay", 0.01)
        betas = tuple(opt_config.get("betas", [0.9, 0.999]))
        
        # 分组参数 (不同层不同学习率)
        param_groups = []
        
        # 编码器参数 (较低学习率)
        encoder_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if "text_encoder" in name or "vision_encoder" in name:
                encoder_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {"params": encoder_params, "lr": lr * 0.1},  # 编码器学习率降低 10 倍
            {"params": other_params, "lr": lr},
        ]
        
        optimizer = AdamW(param_groups, weight_decay=weight_decay, betas=betas)
        
        print(f"优化器：AdamW (lr={lr}, weight_decay={weight_decay})")
        
        return optimizer
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """创建学习率调度器"""
        sched_config = self.config.get("training", {}).get("scheduler", {})
        
        num_epochs = self.config.get("training", {}).get("num_epochs", 10)
        accumulation_steps = self.config.get("training", {}).get("gradient_accumulation_steps", 1)
        warmup_ratio = sched_config.get("warmup_ratio", 0.1)
        # Scheduler steps = optimizer steps (batches / accumulation_steps)
        steps_per_epoch = len(self.train_loader) // accumulation_steps
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = int(total_steps * warmup_ratio)
        
        sched_name = sched_config.get("name", "cosine")
        
        if sched_name == "cosine":
            # Warmup + Cosine
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=sched_config.get("min_lr_ratio", 0.1) * self.config["training"]["optimizer"]["lr"],
            )
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = None
        
        print(f"学习率调度器：{sched_name} (warmup={warmup_ratio})")
        
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # Gradient accumulation and clipping config
        accumulation_steps = self.config.get("training", {}).get("gradient_accumulation_steps", 1)
        max_grad_norm = self.config.get("training", {}).get("max_grad_norm", 1.0)
        
        # 进度条
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}/{self.config['training']['num_epochs']} (Train)",
            leave=False,
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # 准备数据
            input_ids = batch["text_input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            pixel_values = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # 前向传播 + 混合精度
            if self.use_fp16 and self.scaler:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=labels,
                        return_loss=True,
                    )
                    loss = outputs["loss"] / accumulation_steps
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                    return_loss=True,
                )
                loss = outputs["loss"] / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
            
            # 统计 (report un-scaled loss)
            total_loss += loss.item() * accumulation_steps
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})
            
            # TensorBoard 日志
            if self.writer and self.global_step % 10 == 0:
                self.writer.add_scalar("train/loss", loss.item() * accumulation_steps, self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
        
        avg_loss = total_loss / num_batches
        
        return {"train_loss": avg_loss}
    
    @torch.no_grad()
    def evaluate(self, loader: torch.utils.data.DataLoader, split: str = "val") -> Dict[str, float]:
        """评估"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_uncertainties = []
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            loader,
            desc=f"Evaluating ({split})",
            leave=False,
        )
        
        for batch in pbar:
            input_ids = batch["text_input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            pixel_values = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                return_loss=True,
            )
            
            total_loss += outputs["loss"].item()
            
            # 收集预测
            preds = outputs["probs"].argmax(dim=-1).cpu()
            probs = outputs["probs"].cpu()
            uncertainties = outputs["uncertainty"].cpu()
            
            all_preds.append(preds)
            all_labels.append(labels.cpu())
            all_probs.append(probs)
            all_uncertainties.append(uncertainties)
            
            num_batches += 1
        
        # 合并结果
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)
        all_uncertainties = torch.cat(all_uncertainties)
        
        # 计算指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            f"{split}_loss": total_loss / num_batches,
            f"{split}_accuracy": accuracy_score(all_labels, all_preds),
            f"{split}_precision": precision_score(all_labels, all_preds, average="weighted"),
            f"{split}_recall": recall_score(all_labels, all_preds, average="weighted"),
            f"{split}_f1": f1_score(all_labels, all_preds, average="weighted"),
            f"{split}_uncertainty": all_uncertainties.mean().item(),
        }
        
        # AUC (二分类)
        if self.model.num_labels == 2:
            try:
                metrics[f"{split}_auc"] = roc_auc_score(all_labels, all_probs[:, 1])
            except:
                metrics[f"{split}_auc"] = 0.0
        
        return metrics
    
    def save_checkpoint(self, filename: str, extra: Optional[Dict] = None):
        """保存检查点"""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            "config": self.config,
        }
        
        if extra:
            checkpoint.update(extra)
        
        checkpoint_path = checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        print(f"✓ 保存检查点：{checkpoint_path}")
    
    def train(self):
        """完整训练流程"""
        print("=" * 60)
        print("开始训练")
        print("=" * 60)
        
        num_epochs = self.config.get("training", {}).get("num_epochs", 10)
        save_best = self.config.get("evaluation", {}).get("save_best", True)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.evaluate(self.val_loader, "val")
            
            # 测试
            test_metrics = self.evaluate(self.test_loader, "test")
            
            # 时间
            epoch_time = time.time() - start_time
            
            # 合并指标
            all_metrics = {**train_metrics, **val_metrics, **test_metrics, "epoch_time": epoch_time}
            
            # 打印
            print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val  F1:    {val_metrics['val_f1']:.4f}")
            print(f"  Test F1:    {test_metrics['test_f1']:.4f}")
            
            # TensorBoard
            if self.writer:
                for key, value in all_metrics.items():
                    self.writer.add_scalar(key, value, epoch)
            
            # WandB
            if self.wandb_run:
                wandb.log(all_metrics, step=epoch)
            
            # 保存最佳模型
            current_metric = val_metrics.get(f"val_{self.best_metric_name}", 0.0)
            
            is_better = (
                (self.monitor_mode == "max" and current_metric > self.best_metric) or
                (self.monitor_mode == "min" and current_metric < self.best_metric)
            )
            
            if is_better:
                self.best_metric = current_metric
                self.early_stop_counter = 0
                
                if save_best:
                    self.save_checkpoint(
                        f"best_model_epoch{epoch + 1}.pt",
                        extra={"metrics": all_metrics},
                    )
                
                print(f"✨ 新的最佳模型！{self.best_metric_name}={self.best_metric:.4f}")
            else:
                self.early_stop_counter += 1
                
                if self.early_stop_counter >= self.early_stop_patience:
                    print(f"\n🛑 触发早停 (patience={self.early_stop_patience})")
                    break
            
            # 定期保存检查点
            save_freq = self.config.get("logging", {}).get("checkpoint", {}).get("save_freq", 1)
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch + 1}.pt")
        
        # 训练完成
        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"最佳 {self.best_metric_name}: {self.best_metric:.4f}")
        print("=" * 60)
        
        # 关闭日志
        if self.writer:
            self.writer.close()
        
        if self.wandb_run:
            wandb.finish()
        
        return self.best_metric


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    import yaml
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="多模态虚假新闻分类器训练")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--output_dir", type=str, default="results", help="输出目录")
    parser.add_argument("--resume", type=str, default=None, help="恢复检查点路径")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_path = output_dir / "config.yaml"
    import yaml
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"配置已保存到：{config_path}")
    
    # 数据加载
    data_dir = config.get("dataset", {}).get("data_dir", "data/processed/twitter15")
    max_text_length = config.get("dataset", {}).get("max_text_length", 512)
    batch_size = config.get("training", {}).get("batch_size", 8)
    
    print(f"\n加载数据集：{data_dir}")
    
    # 加载分词器
    text_model_name = config.get("model", {}).get("text_encoder", {}).get("name", "FacebookAI/roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    
    # 创建 DataLoader
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        data_dir=data_dir,
        text_tokenizer=tokenizer,
        batch_size=batch_size,
        max_text_length=max_text_length,
        num_workers=config.get("hardware", {}).get("num_workers", 4),
        use_weighted_sampling=True,
    )
    
    # 创建模型
    print(f"\n创建模型...")
    
    model_config = config.get("model", {})
    model = MultimodalFakeNewsClassifier(
        text_model_name=model_config.get("text_encoder", {}).get("name", "FacebookAI/roberta-base"),
        vision_model_name=model_config.get("vision_encoder", {}).get("name", "google/vit-base-patch16-224"),
        num_labels=2,
        moe_num_experts=model_config.get("moe_lora", {}).get("num_experts", 3),
        moe_lora_r=model_config.get("moe_lora", {}).get("lora_r", 8),
        moe_lora_alpha=model_config.get("moe_lora", {}).get("lora_alpha", 16.0),
        freeze_encoders=False,
    )
    
    # 参数量统计
    param_stats = model.get_num_params()
    print(f"模型参数量:")
    print(f"  总计：{param_stats['total']:,}")
    print(f"  可训练：{param_stats['trainable']:,}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        output_dir=str(output_dir),
    )
    
    # 恢复检查点 (可选)
    if args.resume:
        print(f"\n恢复检查点：{args.resume}")
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.epoch = checkpoint["epoch"]
        trainer.global_step = checkpoint["global_step"]
    
    # 开始训练
    trainer.train()
    
    print(f"\n训练完成！结果保存在：{output_dir}")


if __name__ == "__main__":
    main()
