#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 微调训练脚本

包含三个子命令：train（训练）、merge（合并）、deploy（部署）
"""

import argparse
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LoRA 微调训练脚本")
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # train 子命令
    train_parser = subparsers.add_parser("train", help="训练 LoRA 适配器")
    train_parser.add_argument("--base-model", required=True, help="HuggingFace 模型 ID 或本地路径")
    train_parser.add_argument("--data-dir", required=True, help="训练数据目录")
    train_parser.add_argument("--output-dir", required=True, help="输出目录")
    train_parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    train_parser.add_argument("--batch-size", type=int, default=2, help="批量大小")
    train_parser.add_argument("--grad-accum", type=int, default=8, help="梯度累积步数")
    train_parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    train_parser.add_argument("--lora-r", type=int, default=16, help="LoRA 秩")
    train_parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    train_parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    train_parser.add_argument("--max-seq-len", type=int, default=4096, help="最大序列长度")
    train_parser.add_argument("--eval-steps", type=int, default=100, help="评估步数")
    train_parser.add_argument("--save-steps", type=int, default=200, help="保存步数")
    train_parser.add_argument("--no-4bit", action="store_true", help="使用 bf16 全精度替代 QLoRA")
    train_parser.add_argument("--resume-adapter", help="从已有 adapter 继续训练")
    train_parser.add_argument("--resume-checkpoint", help="从训练 checkpoint 恢复")
    
    # merge 子命令
    merge_parser = subparsers.add_parser("merge", help="合并 LoRA 适配器到基础模型")
    merge_parser.add_argument("--base-model", required=True, help="基础模型路径")
    merge_parser.add_argument("--adapter-path", required=True, help="LoRA 适配器路径")
    merge_parser.add_argument("--output-path", required=True, help="合并后模型输出路径")
    
    # deploy 子命令
    deploy_parser = subparsers.add_parser("deploy", help="部署微调后的模型")
    deploy_parser.add_argument("--model-path", required=True, help="模型路径")
    deploy_parser.add_argument("--config-file", help="部署配置文件")
    
    return parser.parse_args()


def load_training_dataset(data_dir):
    """加载训练数据集"""
    train_file = os.path.join(data_dir, "train.jsonl")
    val_file = os.path.join(data_dir, "val.jsonl")
    
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    val_dataset = load_dataset("json", data_files=val_file, split="train")
    
    return train_dataset, val_dataset


def train(args):
    """训练 LoRA 适配器"""
    print(f"开始训练 LoRA 适配器...")
    print(f"基础模型: {args.base_model}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    
    # 加载数据集
    train_dataset, val_dataset = load_training_dataset(args.data_dir)
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 配置模型加载
    if not args.no_4bit:
        # 使用 4bit QLoRA
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quantization_config,
            device_map="auto"
        )
    else:
        # 使用 bf16 全精度
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    # 配置 LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    
    # 从已有 adapter 继续训练
    if args.resume_adapter:
        model.load_adapter(args.resume_adapter)
        print(f"从已有适配器继续训练: {args.resume_adapter}")
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=not args.no_4bit,
        tf32=False,
        push_to_hub=False,
        report_to="tensorboard"
    )
    
    # 定义数据处理函数
    def process_func(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        example["text"] = text
        return example
    
    # 处理数据集
    train_dataset = train_dataset.map(process_func)
    val_dataset = val_dataset.map(process_func)
    
    # 创建 SFT 训练器
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        packing=False,
        max_seq_length=args.max_seq_len
    )
    
    # 开始训练
    if args.resume_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_checkpoint)
    else:
        trainer.train()
    
    # 保存模型
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"训练完成！模型保存到: {args.output_dir}")


def merge(args):
    """合并 LoRA 适配器到基础模型"""
    print(f"开始合并 LoRA 适配器...")
    print(f"基础模型: {args.base_model}")
    print(f"适配器路径: {args.adapter_path}")
    print(f"输出路径: {args.output_path}")
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # 加载 LoRA 适配器
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    
    # 合并模型
    merged_model = peft_model.merge_and_unload()
    
    # 保存合并后的模型
    merged_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    
    print(f"合并完成！模型保存到: {args.output_path}")


def deploy(args):
    """部署微调后的模型"""
    print(f"开始部署模型...")
    print(f"模型路径: {args.model_path}")
    
    # 这里可以添加部署逻辑，例如：
    # 1. 加载模型到内存
    # 2. 启动推理服务
    # 3. 更新配置文件
    
    # 示例：验证模型是否可以正常加载
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        print("模型加载成功！")
        print(f"模型类型: {type(model).__name__}")
        print(f"Tokenizer 类型: {type(tokenizer).__name__}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 如果提供了配置文件，更新配置
    if args.config_file:
        print(f"更新配置文件: {args.config_file}")
        # 这里可以添加更新配置文件的逻辑
    
    print("部署完成！")


def main():
    """主函数"""
    args = parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "merge":
        merge(args)
    elif args.command == "deploy":
        deploy(args)
    else:
        print("请指定子命令: train, merge, deploy")


if __name__ == "__main__":
    main()
