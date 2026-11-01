import os
import torch
import numpy as np
import mujoco
import mujoco.viewer
from envs.base_env import BaseLeggedEnv
from algos.ppo import PPOAgent
from algos.utils import discount_rewards


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 当前训练设备: {device}")

    xml_path = os.path.join("..", "assets", "unitree_go2", "scene.xml")
    env = BaseLeggedEnv(xml_path)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim, action_dim, device)

    save_dir = os.path.join("..", "pretrain")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model_spine.pth")

    # 保留：在原有权重上继续加载与训练的功能
    if os.path.exists(save_path):
        agent.policy.load_state_dict(torch.load(save_path, map_location=device))
        print(f"✅ 成功加载已保存权重: {save_path}，继续累加训练！")
    else:
        print("🆕 未找到历史权重，从头开始训练。")

    total_episodes = 50000000
    gamma = 0.99

    # 启动实时可视化 3D 渲染窗口
    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    print("🖥️ 实时可视化窗口已开启！(训练速度可能会变慢)")

    try:
        for ep in range(total_episodes):
            if not viewer.is_running():
                print("检测到渲染窗口被关闭，退出训练。")
                break

            # 扩展课程学习阶段：适配上下坡、上下楼梯、梅花桩及终点墙壁冲刺
            # 阶段 1：平地与上下坡
            # 阶段 2：上下楼梯
            # 阶段 3：梅花桩跳跃
            # 阶段 4：全地形综合与终点墙壁冲刺
            if ep < 1500:
                target_stage = 1
            elif ep < 4000:
                target_stage = 2
            elif ep < 8000:
                target_stage = 3
            else:
                target_stage = 4

            if env.curriculum_stage != target_stage:
                env.set_curriculum_stage(target_stage)

            obs, _ = env.reset()
            done = False

            memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': []}
            episode_reward = 0

            while not done:
                if not viewer.is_running():
                    break

                action, log_prob, val = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                memory['states'].append(obs)
                memory['actions'].append(action)
                memory['log_probs'].append(log_prob)
                memory['rewards'].append(reward)
                memory['values'].append(val)
                memory['dones'].append(done)

                obs = next_obs
                episode_reward += reward

                # 同步画面到 3D 渲染窗口
                viewer.sync()

            if not viewer.is_running():
                break

            # 更新状态归一化统计量
            states_tensor = torch.FloatTensor(np.array(memory['states'])).to(device)
            agent.policy.obs_normalizer.update(states_tensor)

            memory['returns'] = discount_rewards(memory['rewards'], memory['dones'], gamma)
            returns_t = torch.FloatTensor(memory['returns']).to(device)
            values_t = torch.FloatTensor(memory['values']).to(device)
            memory['advantages'] = (returns_t - values_t).detach().cpu().tolist()

            agent.update(memory)

            final_y = env.data.qpos[1]

            # 动态推进课程阶段（根据机器狗在Y轴向前移动的距离自动解锁更高难度地形）
            if env.curriculum_stage == 1 and final_y >= 3.0:
                env.set_curriculum_stage(2)
            elif env.curriculum_stage == 2 and final_y >= 8.0:
                env.set_curriculum_stage(3)
            elif env.curriculum_stage == 3 and final_y >= 16.0:
                env.set_curriculum_stage(4)

            print(
                f"Episode {ep + 1}/{total_episodes} | 阶段: Stage {env.curriculum_stage} | 最终Y: {final_y:.2f} | 奖励: {episode_reward:.2f}")

            # 每 100 轮自动保存一次模型权重
            if (ep + 1) % 100 == 0:
                torch.save(agent.policy.state_dict(), save_path)

    finally:
        viewer.close()
        torch.save(agent.policy.state_dict(), save_path)
        print(f"🎉 训练中断/完成，权重已保存至 {save_path}")


if __name__ == "__main__":
    main()