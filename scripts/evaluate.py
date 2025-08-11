#!/usr/bin/env python3
"""
Causal-MAC 模型评估入口脚本
使用示例: 
  python evaluate.py --model checkpoints/best_model.pt --env-config configs/env/custom_map.yaml
  python evaluate.py --compare --baseline ic3net --num-episodes 100
"""

import argparse
import os
import json
import torch
from evaluation.evaluator import CausalMACEvaluator
from evaluation.benchmark import BenchmarkComparator
from configs import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Causal-MAC 评估脚本")
    parser.add_argument("--model", type=str, required=False,
                        help="待评估模型路径")
    parser.add_argument("--env-config", type=str, default="configs/env/pursuit_v4.yaml", 
                        help="环境配置文件路径")
    parser.add_argument("--model-config", type=str, default="configs/model/causal_mac.yaml", 
                        help="模型配置文件路径")
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="评估回合数")
    parser.add_argument("--compare", action="store_true",
                        help="与基准模型对比")
    parser.add_argument("--baseline", type=str, choices=["ic3net", "tarmac", "all"], default="all",
                        help="选择对比的基准模型")
    parser.add_argument("--output", type=str, default="results/evaluation.json",
                        help="评估结果输出路径")
    parser.add_argument("--report", type=str, default="results/report.txt",
                        help="文本报告输出路径")
    parser.add_argument("--plot", type=str, default="results/comparison.png",
                        help="对比图表输出路径")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="评估设备")
    return parser.parse_args()

def load_model(model_path, env_config, model_config, device):
    """加载模型"""
    from models import AgentPolicy
    
    # 初始化模型
    model = AgentPolicy(
        obs_dim=env_config["obs_dim"],
        action_dim=env_config["action_dim"],
        config=model_config
    )
    
    # 加载权重
    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict["policy_state_dict"] if "policy_state_dict" in state_dict else state_dict)
    
    model.to(device)
    model.eval()
    return model

def main():
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    
    # 加载配置文件
    env_config = load_config(args.env_config)
    model_config = load_config(args.model_config)
    
    # 设置环境观测和动作维度
    env_config["obs_dim"] = 30  # 根据TinyScenes特征维度设置
    env_config["action_dim"] = 5  # 追捕环境的动作空间
    
    # 单一模型评估
    if not args.compare:
        if not args.model:
            raise ValueError("评估单一模型时必须提供 --model 参数")
        
        print(f"评估模型: {args.model}")
        model = load_model(args.model, env_config, model_config, args.device)
        
        # 初始化评估器
        evaluator = CausalMACEvaluator(
            policy_model=model,
            env_config=env_config,
            model_config=model_config,
            device=args.device
        )
        
        # 执行评估
        metrics = evaluator.evaluate(args.num_episodes)
        
        # 保存结果
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # 生成报告
        report = evaluator.generate_report()
        with open(args.report, "w") as f:
            f.write(report)
        
        print(f"评估完成! 结果保存至: {args.output}")
        print(report)
    
    # 基准模型对比
    else:
        print("执行基准模型对比评估...")
        
        # 加载Causal-MAC模型
        causal_mac = load_model(args.model, env_config, model_config, args.device)
        
        # 初始化对比器
        comparator = BenchmarkComparator(
            causal_mac_model=causal_mac,
            env_config=env_config,
            model_config=model_config,
            device=args.device
        )
        
        # 执行对比评估
        results = comparator.run_comparison(args.num_episodes)
        
        # 保存结果
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        
        # 生成报告和图表
        report = comparator.generate_comparison_report(results)
        with open(args.report, "w") as f:
            f.write(report)
        
        fig = comparator.plot_performance_comparison(results, save_path=args.plot)
        
        print(f"对比评估完成! 结果保存至: {args.output}")
        print(f"报告保存至: {args.report}")
        print(f"图表保存至: {args.plot}")
        print(report)

if __name__ == "__main__":
    main()