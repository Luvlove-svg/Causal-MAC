import torch
import torch.nn as nn
import numpy as np
from bitsandbytes import functional as F

def quantize_layer_int4(linear_layer):
    """将线性层转换为INT4量化格式"""
    # 保存原始权重和偏置
    original_weight = linear_layer.weight.data.clone()
    original_bias = linear_layer.bias.data.clone() if linear_layer.bias is not None else None
    
    # 量化权重
    quant_weight, quant_state = F.quantize_fp4(original_weight)
    
    # 替换前向传播
    def quantized_forward(x):
        # 反量化权重
        dequant_weight = F.dequantize_fp4(quant_weight, quant_state)
        return nn.functional.linear(x, dequant_weight, original_bias)
    
    linear_layer.forward = quantized_forward
    return linear_layer

class QuantizationAwareTraining(nn.Module):
    """量化感知训练包装器"""
    def __init__(self, model, quantize_after_epoch=5):
        """
        Args:
            model: 待量化模型
            quantize_after_epoch: 开始量化的训练轮次
        """
        super().__init__()
        self.model = model
        self.quantize_after_epoch = quantize_after_epoch
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        """设置当前训练轮次"""
        self.current_epoch = epoch
    
    def forward(self, *args, **kwargs):
        """带量化模拟的前向传播"""
        if self.training and self.current_epoch >= self.quantize_after_epoch:
            # 在训练后期模拟量化效果
            with torch.no_grad():
                output = self.model(*args, **kwargs)
            return output
        else:
            # 普通前向传播
            return self.model(*args, **kwargs)