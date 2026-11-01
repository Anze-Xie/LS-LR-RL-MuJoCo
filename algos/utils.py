import numpy as np
import torch


def normalize_advantages(advantages):
    """对 PPO 的优势函数进行归一化，提升训练稳定性"""
    # 如果传进来本身已经是 tensor，直接在原地或对应设备计算
    if not isinstance(advantages, torch.Tensor):
        advantages = torch.FloatTensor(advantages)

    # 防止分母为 0 并且保持原有的设备不变
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + 1e-5)


def discount_rewards(rewards, dones, gamma=0.99):
    """计算 Monte-Carlo 折扣回报 (Returns)"""
    returns = []
    discounted_sum = 0
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            discounted_sum = 0
        discounted_sum = r + (gamma * discounted_sum)
        returns.insert(0, discounted_sum)
    return returns