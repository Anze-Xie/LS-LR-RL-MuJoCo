import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from algos.actor_critic import ActorCritic
from algos.utils import normalize_advantages

class PPOAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        # 传入 device 并在 GPU 上初始化网络
        self.policy = ActorCritic(state_dim, action_dim, device).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        # 关键修复：添加 .to(self.device)，将输入状态张量推送到 GPU
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, action_logprob, state_val = self.policy.act(state_t)
        # 将输出转回 CPU 转换为 numpy
        return action.squeeze(0).cpu().numpy(), action_logprob.item(), state_val.item()

    def update(self, memory):
        # 确保训练更新时的所有张量也在 GPU 上
        states = torch.FloatTensor(np.array(memory['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(memory['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(memory['log_probs'])).to(self.device)
        returns = torch.FloatTensor(np.array(memory['returns'])).to(self.device)
        advantages = torch.FloatTensor(np.array(memory['advantages'])).to(self.device)

        advantages = normalize_advantages(advantages).to(self.device)

        for _ in range(4):
            log_probs, state_values, entropy = self.policy.evaluate(states, actions)
            state_values = state_values.squeeze(-1)

            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - 0.2, 1.0 + 0.2) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse_loss(state_values, returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()