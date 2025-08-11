#!/usr/bin/env python3
"""
Causal-MAC 结果可视化脚本
使用示例: 
  python visualize.py --log training_log.json --type training
  python visualize.py --log evaluation.json --type metrics
  python visualize.py --comm comm_log.json
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def parse_args():
    parser = argparse.ArgumentParser(description="Causal-MAC 可视化工具")
    parser.add_argument("--log", type=str, required=True,
                        help="日志文件路径 (训练日志或评估结果)")
    parser.add_argument("--type", type=str, choices=["training", "metrics"], default="training",
                        help="日志类型 (训练日志或评估指标)")
    parser.add_argument("--comm", type=str,
                        help="通信日志文件路径 (用于通信分析)")
    parser.add_argument("--output-dir", type=str, default="results/plots",
                        help="图表输出目录")
    return parser.parse_args()

def plot_training_curves(log_data, output_dir):
    """绘制训练曲线"""
    rewards = log_data["rewards"]
    losses = log_data.get("losses", [])
    steps = log_data.get("steps", list(range(len(rewards))))
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 奖励曲线
    ax1.plot(steps, rewards, 'b-', linewidth=1.5)
    ax1.set_title('训练奖励曲线')
    ax1.set_ylabel('平均回合奖励')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 损失曲线
    if losses:
        ax2.plot(steps, losses, 'r-', linewidth=1.5)
        ax2.set_title('训练损失曲线')
        ax2.set_xlabel('训练步数')
        ax2.set_ylabel('损失值')
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(output_path, dpi=300)
    print(f"训练曲线保存至: {output_path}")
    
    return fig

def plot_metrics_comparison(metrics_data, output_dir):
    """绘制指标对比图"""
    if not isinstance(metrics_data, dict) or "causal_mac" not in metrics_data:
        raise ValueError("无效的指标数据格式")
    
    # 提取指标
    models = list(metrics_data.keys())
    metrics = ["success_rate", "avg_reward", "invalid_message_ratio"]
    labels = ["成功率", "平均奖励", "无效消息比例"]
    
    values = {}
    for metric in metrics:
        values[metric] = [metrics_data[model][metric] for model in models]
    
    # 创建图表
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(models, values[metric], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_title(labels[i])
        ax.set_ylabel(metric)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.suptitle('模型性能对比', fontsize=16)
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"指标对比图保存至: {output_path}")
    
    return fig

def plot_communication_analysis(comm_data, output_dir):
    """分析并可视化通信模式"""
    if not comm_data or "steps" not in comm_data:
        raise ValueError("无效的通信日志格式")
    
    steps = comm_data["steps"]
    total_messages = []
    effective_messages = []
    message_lengths = []
    
    for step in steps:
        total = len(step)
        effective = sum(1 for comm in step if comm["effective"])
        avg_length = np.mean([len(comm["message"]) for comm in step]) if step else 0
        
        total_messages.append(total)
        effective_messages.append(effective)
        message_lengths.append(avg_length)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 消息数量
    ax1.plot(total_messages, 'b-', label='总消息数')
    ax1.plot(effective_messages, 'g-', label='有效消息数')
    ax1.set_title('消息数量随时间变化')
    ax1.set_ylabel('消息数量')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 消息长度
    ax2.plot(message_lengths, 'r-')
    ax2.set_title('平均消息长度随时间变化')
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('平均消息长度')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, "communication_analysis.png")
    plt.savefig(output_path, dpi=300)
    print(f"通信分析图保存至: {output_path}")
    
    return fig

def plot_causal_usage(comm_data, causal_graph, output_dir):
    """可视化因果图使用情况"""
    if not comm_data or "steps" not in comm_data:
        raise ValueError("无效的通信日志格式")
    
    # 统计边使用次数
    edge_usage = {edge: 0 for edge in causal_graph["edges"]}
    
    for step in comm_data["steps"]:
        for comm in step:
            edge = (comm["sender"], comm["receiver"])
            if edge in edge_usage:
                edge_usage[edge] += 1
    
    # 准备绘图数据
    edges = list(edge_usage.keys())
    usages = list(edge_usage.values())
    labels = [f"{s}→{r}" for s, r in edges]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, usages, color='skyblue')
    
    ax.set_title('因果边使用情况')
    ax.set_ylabel('使用次数')
    ax.set_xlabel('因果边 (发送者→接收者)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, "causal_edge_usage.png")
    plt.savefig(output_path, dpi=300)
    print(f"因果边使用图保存至: {output_path}")
    
    return fig

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载日志数据
    with open(args.log, 'r') as f:
        log_data = json.load(f)
    
    # 根据类型绘制图表
    if args.type == "training":
        plot_training_curves(log_data, args.output_dir)
    
    elif args.type == "metrics":
        plot_metrics_comparison(log_data, args.output_dir)
    
    # 通信分析
    if args.comm:
        with open(args.comm, 'r') as f:
            comm_data = json.load(f)
        
        plot_communication_analysis(comm_data, args.output_dir)
        
        # 如果有因果图数据，绘制因果边使用情况
        if "causal_graph" in comm_data:
            plot_causal_usage(comm_data, comm_data["causal_graph"], args.output_dir)

if __name__ == "__main__":
    main()