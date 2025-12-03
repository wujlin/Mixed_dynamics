"""
情绪分类器模块 (Emotion Classifier)

基于 HuggingFace Transformers 训练轻量级分类器。

使用方法：
    # 训练
    classifier = EmotionClassifier()
    classifier.train(train_texts, train_labels)
    
    # 推理
    predictions = classifier.predict(texts)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClassifierConfig:
    """分类器配置"""
    model_name: str = "bert-base-chinese"  # 或 "hfl/chinese-roberta-wwm-ext"
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    output_dir: str = "outputs/classifier"


class EmotionClassifier:
    """
    情绪分类器
    
    支持两个分类任务：
    1. emotion_class: H/M/L
    2. risk_class: risk/norisk
    
    Parameters
    ----------
    config : ClassifierConfig
        配置参数
    """
    
    EMOTION_LABELS = ["H", "M", "L"]
    RISK_LABELS = ["risk", "norisk"]
    
    def __init__(self, config: Optional[ClassifierConfig] = None):
        self.config = config or ClassifierConfig()
        self.emotion_model = None
        self.risk_model = None
        self.tokenizer = None
        self._initialized = False
    
    def _init_models(self):
        """延迟初始化模型（避免不必要的加载）"""
        if self._initialized:
            return
        
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            logger.info(f"加载分词器: {self.config.model_name}")
            self._initialized = True
            
        except ImportError:
            raise ImportError(
                "请安装 transformers: pip install transformers torch"
            )
    
    def train(
        self,
        texts: List[str],
        emotion_labels: List[str],
        risk_labels: Optional[List[str]] = None,
        val_split: float = 0.1,
    ) -> Dict[str, Any]:
        """
        训练分类器
        
        Parameters
        ----------
        texts : List[str]
            训练文本
        emotion_labels : List[str]
            情绪标签 (H/M/L)
        risk_labels : List[str], optional
            风险标签 (risk/norisk)
        val_split : float
            验证集比例
            
        Returns
        -------
        Dict[str, Any]
            训练结果
        """
        self._init_models()
        
        from transformers import (
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
        from sklearn.model_selection import train_test_split
        import torch
        
        # 准备数据
        emotion_label_map = {label: i for i, label in enumerate(self.EMOTION_LABELS)}
        emotion_ids = [emotion_label_map[l] for l in emotion_labels]
        
        # 分割数据
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, emotion_ids, test_size=val_split, random_state=42
        )
        
        # 创建数据集
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.encodings = tokenizer(
                    texts, truncation=True, padding=True, max_length=max_length
                )
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item
            
            def __len__(self):
                return len(self.labels)
        
        train_dataset = TextDataset(train_texts, train_labels, self.tokenizer, self.config.max_length)
        val_dataset = TextDataset(val_texts, val_labels, self.tokenizer, self.config.max_length)
        
        # 初始化模型
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.EMOTION_LABELS),
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_ratio=self.config.warmup_ratio,
            learning_rate=self.config.learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=50,
        )
        
        # 训练
        trainer = Trainer(
            model=self.emotion_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        train_result = trainer.train()
        eval_result = trainer.evaluate()
        
        return {
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result["eval_loss"],
            "train_samples": len(train_texts),
            "val_samples": len(val_texts),
        }
    
    def predict(
        self,
        texts: List[str],
        task: str = "emotion",
    ) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Parameters
        ----------
        texts : List[str]
            待预测文本
        task : str
            任务类型 ("emotion" 或 "risk")
            
        Returns
        -------
        List[Dict[str, Any]]
            预测结果，包含 label 和 confidence
        """
        self._init_models()
        
        import torch
        
        model = self.emotion_model if task == "emotion" else self.risk_model
        labels = self.EMOTION_LABELS if task == "emotion" else self.RISK_LABELS
        
        if model is None:
            raise ValueError(f"模型未训练或加载: {task}")
        
        model.eval()
        results = []
        
        # 批量处理
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i : i + self.config.batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                confs = probs.max(dim=-1).values
            
            for pred, conf in zip(preds.tolist(), confs.tolist()):
                results.append({
                    "label": labels[pred],
                    "confidence": conf,
                })
        
        return results
    
    def save(self, path: str):
        """保存模型"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.emotion_model:
            self.emotion_model.save_pretrained(path / "emotion")
        if self.risk_model:
            self.risk_model.save_pretrained(path / "risk")
        if self.tokenizer:
            self.tokenizer.save_pretrained(path / "tokenizer")
        
        # 保存配置
        with open(path / "config.json", "w") as f:
            json.dump(vars(self.config), f)
        
        logger.info(f"模型已保存到: {path}")
    
    def load(self, path: str):
        """加载模型"""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        path = Path(path)
        
        if (path / "tokenizer").exists():
            self.tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        
        if (path / "emotion").exists():
            self.emotion_model = AutoModelForSequenceClassification.from_pretrained(path / "emotion")
        
        if (path / "risk").exists():
            self.risk_model = AutoModelForSequenceClassification.from_pretrained(path / "risk")
        
        if (path / "config.json").exists():
            with open(path / "config.json") as f:
                config_dict = json.load(f)
                self.config = ClassifierConfig(**config_dict)
        
        self._initialized = True
        logger.info(f"模型已从 {path} 加载")


def evaluate_classifier(
    predictions: List[Dict[str, Any]],
    ground_truth: List[str],
) -> Dict[str, float]:
    """
    评估分类器性能
    
    Parameters
    ----------
    predictions : List[Dict]
        预测结果
    ground_truth : List[str]
        真实标签
        
    Returns
    -------
    Dict[str, float]
        评估指标
    """
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
    
    pred_labels = [p["label"] for p in predictions]
    
    return {
        "accuracy": accuracy_score(ground_truth, pred_labels),
        "f1_macro": f1_score(ground_truth, pred_labels, average="macro"),
        "f1_weighted": f1_score(ground_truth, pred_labels, average="weighted"),
        "cohen_kappa": cohen_kappa_score(ground_truth, pred_labels),
    }

