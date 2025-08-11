import torch
import numpy as np

def hard_update(target, source):
    """硬更新目标网络参数"""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)

def soft_update(target, source, tau):
    """软更新目标网络参数"""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_param.data * tau
        )

def to_tensor(data, device='cuda'):
    """将数据转换为张量"""
    if isinstance(data, dict):
        return {k: to_tensor(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [to_tensor(x, device) for x in data]
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, device=device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def log_metrics(metrics, step, writer):
    """记录训练指标到TensorBoard"""
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                writer.add_scalar(f"{key}/{sub_key}", sub_value, step)
        else:
            writer.add_scalar(key, value, step)

def compute_grad_norm(model):
    """计算模型梯度的范数"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def set_requires_grad(model, requires_grad):
    """设置模型参数的梯度要求"""
    for param in model.parameters():
        param.requires_grad = requires_grad

def freeze_model(model):
    """冻结模型参数"""
    set_requires_grad(model, False)

def unfreeze_model(model):
    """解冻模型参数"""
    set_requires_grad(model, True)