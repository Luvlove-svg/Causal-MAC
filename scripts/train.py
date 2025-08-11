#!/usr/bin/env python3
"""
Causal-MAC 模型训练入口脚本
使用示例: python train.py --config configs/training.yaml --env-config configs/env/pursuit_v4.yaml
"""

import argparse
import os
import yaml
import torch
from training.trainer import CausalMACTrainer
from configs import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Causal-MAC 训练脚本")
    parser.add_argument("--config", type=str, default="configs/training.yaml", 
                        help="主训练配置文件路径")
    parser.add_argument("--env-config", type=str, default="configs/env/pursuit_v4.yaml", 
                        help="环境配置文件路径")
    parser.add_argument("--model-config", type=str, default="configs/model/causal_mac.yaml", 
                        help="模型配置文件路径")
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点恢复训练，提供检查点路径")
    parser.add_argument("--output-dir", type=str, default="checkpoints/",
                        help="模型检查点输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="训练设备")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置文件
    config = load_config(args.config)
    env_config = load_config(args.env_config)
    model_config = load_config(args.model_config)
    
    # 设置硬件配置
    config["hardware"] = {
        "causal_discovery_cores": 24,  # 使用全部24核
        "simulation_cores": "0-5",     # 绑定到前6核
        "gpu_id": 0 if args.device == "cuda" else None
    }
    
    # 初始化训练器
    trainer = CausalMACTrainer(config, env_config, model_config, device=args.device)
    
    # 从检查点恢复
    if args.resume:
        print(f"从检查点恢复训练: {args.resume}")
        trainer.load(args.resume)
    
    # 启动训练
    print(f"开始训练，配置: {args.config}")
    print(f"环境: {args.env_config}")
    print(f"模型: {args.model_config}")
    print(f"设备: {args.device}")
    
    rewards = trainer.run()
    
    # 保存最终模型
    final_path = os.path.join(args.output_dir, "final_model.pt")
    trainer.save(final_path)
    print(f"训练完成，最终模型保存至: {final_path}")

if __name__ == "__main__":
    main()