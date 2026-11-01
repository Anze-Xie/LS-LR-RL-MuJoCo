import os
import torch
import mujoco
import mujoco.viewer
from envs.base_env import BaseLeggedEnv
from algos.ppo import PPOAgent


def main():
    # 自动选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 评估演练运行设备: {device}")

    xml_path = os.path.join("..", "assets", "unitree_go2", "scene.xml")
    if not os.path.exists(xml_path):
        print(f"提示: 未在 {xml_path} 找到模型文件！")
        return

    env = BaseLeggedEnv(xml_path)

    # 传入 state_dim, action_dim, device
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], device)
    weight_path = os.path.join("..", "pretrain", "best_model_spine.pth")

    if os.path.exists(weight_path):
        # 兼容性加载权重到对应设备
        agent.policy.load_state_dict(torch.load(weight_path, map_location=device))
        print("✅ 已成功加载训练好的策略权重，开始可视化演练！")
    else:
        print("⚠️ 未找到权重文件，将使用随机动作进行演示。")

    # 启动 3D 图形化渲染窗口
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        obs, _ = env.reset()
        while viewer.is_running():
            # 策略选择动作
            action, _, _ = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)

            # 同步渲染窗口画面
            viewer.sync()

            # 如果机器狗摔倒或完成单局，自动重置环境
            if terminated or truncated:
                obs, _ = env.reset()


if __name__ == "__main__":
    main()