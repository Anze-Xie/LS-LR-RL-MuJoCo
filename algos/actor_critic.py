import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    def __init__(self, shape, device):
        super().__init__()
        # 使用 register_buffer 确保均值和方差能随 state_dict 一起保存和加载
        self.register_buffer("mean", torch.zeros(shape, device=device))
        self.register_buffer("var", torch.ones(shape, device=device))
        self.register_buffer("count", torch.tensor(1e-4, device=device))

    def update(self, x):
        if x.numel() == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super().__init__()
        self.device = device
        self.obs_normalizer = RunningMeanStd(state_dim, device)

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim, device=device))

    def act(self, state):
        normalized_state = self.obs_normalizer.normalize(state)
        mean = self.actor(normalized_state)
        std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        state_val = self.critic(normalized_state)
        return action, action_logprob, state_val

    def evaluate(self, state, action):
        normalized_state = self.obs_normalizer.normalize(state)
        mean = self.actor(normalized_state)
        std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean, std)
        action_logprob = dist.log_prob(action).sum(dim=-1)
        state_val = self.critic(normalized_state)
        return action_logprob, state_val, dist.entropy().mean()