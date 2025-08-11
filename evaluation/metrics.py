import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_communication_efficiency(communication_log):
    """
    计算通信效率指标
    Args:
        communication_log: 包含每步通信记录的列表
    Returns:
        效率指标字典
    """
    total_messages = 0
    effective_messages = 0
    message_lengths = []
    
    for step_log in communication_log:
        for comm in step_log:
            total_messages += 1
            if comm['effective']:
                effective_messages += 1
            message_lengths.append(len(comm['message']))
    
    metrics = {
        'message_per_step': total_messages / len(communication_log) if communication_log else 0,
        'effective_ratio': effective_messages / total_messages if total_messages > 0 else 0,
        'avg_message_length': np.mean(message_lengths) if message_lengths else 0,
        'redundancy_ratio': 1 - (effective_messages / total_messages) if total_messages > 0 else 0
    }
    
    return metrics

def calculate_collaboration_metrics(actions, optimal_actions):
    """
    计算协作效率指标
    Args:
        actions: 实际动作序列 [episodes, steps, agents]
        optimal_actions: 最优动作序列 [episodes, steps, agents]
    Returns:
        协作指标字典
    """
    if len(actions) != len(optimal_actions) or len(actions) == 0:
        return {}
    
    episodes = len(actions)
    steps = len(actions[0])
    agents = len(actions[0][0])
    
    # 计算动作准确率
    correct_actions = 0
    total_actions = episodes * steps * agents
    
    for ep in range(episodes):
        for step in range(steps):
            for agent in range(agents):
                if actions[ep][step][agent] == optimal_actions[ep][step][agent]:
                    correct_actions += 1
    
    action_accuracy = correct_actions / total_actions
    
    # 计算协作一致性
    collaboration_consistency = 0
    for ep in range(episodes):
        for step in range(steps):
            agent_actions = actions[ep][step]
            if len(set(agent_actions)) == 1:  # 所有智能体执行相同动作
                collaboration_consistency += 1
    
    collaboration_consistency /= (episodes * steps)
    
    return {
        'action_accuracy': action_accuracy,
        'collaboration_consistency': collaboration_consistency
    }

def calculate_causal_effectiveness(causal_graph, communication_log):
    """
    计算因果推理的有效性
    Args:
        causal_graph: 因果图数据结构
        communication_log: 通信日志
    Returns:
        因果有效性指标
    """
    causal_edges_used = 0
    total_causal_edges = causal_graph.edge_count()
    total_communications = 0
    
    for step_log in communication_log:
        for comm in step_log:
            total_communications += 1
            if causal_graph.has_edge(comm['sender'], comm['receiver']):
                causal_edges_used += 1
    
    return {
        'causal_edge_usage': causal_edges_used / total_causal_edges if total_causal_edges > 0 else 0,
        'causal_decision_ratio': causal_edges_used / total_communications if total_communications > 0 else 0
    }

def calculate_resource_utilization(gpu_mem_log, cpu_usage_log):
    """
    计算资源利用率指标
    Args:
        gpu_mem_log: GPU显存使用记录 (MB)
        cpu_usage_log: CPU使用率记录 (%)
    Returns:
        资源指标字典
    """
    if not gpu_mem_log or not cpu_usage_log:
        return {}
    
    return {
        'avg_gpu_mem': np.mean(gpu_mem_log),
        'max_gpu_mem': np.max(gpu_mem_log),
        'avg_cpu_usage': np.mean(cpu_usage_log),
        'max_cpu_usage': np.max(cpu_usage_log)
    }