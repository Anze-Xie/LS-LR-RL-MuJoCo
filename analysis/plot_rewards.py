import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve():
    # 示例：假设你从训练日志中读取了每一轮的总奖励
    # 实际使用时可将其替换为从文件（如 CSV 或文本日志）中加载数据
    episodes = np.arange(1, 501)
    # 模拟平滑的奖励上升趋势
    rewards = 50 + 200 * (1 - np.exp(-episodes / 100)) + np.random.normal(0, 15, size=500)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label='Episode Reward', color='blue', alpha=0.6)

    # 计算滑动平均曲线
    window_size = 20
    smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    plt.plot(episodes[window_size - 1:], smoothed_rewards, label=f'Moving Average (window={window_size})', color='red',
             linewidth=2)

    plt.title('PPO Training Reward Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_training_curve()