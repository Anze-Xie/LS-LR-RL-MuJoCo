import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are


class BaseLeggedEnv(gym.Env):
    def __init__(self, xml_path):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.n_substeps = 5
        self.curriculum_stage = 1

        self.action_dim = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # ==================== 🧮 LQR 控制器求解配置 ====================
        # 单关节近似转动惯量与物理阻尼
        I = 0.05  # 关节转动惯量 kg·m²
        D = 0.5  # 物理阻尼 N·s/m

        # 连续时间状态空间矩阵 A, B
        A = np.array([[0.0, 1.0],
                      [0.0, -D / I]])
        B = np.array([[0.0],
                      [1.0 / I]])

        # 设定 LQR 权重矩阵 (Q 惩罚状态偏差，R 惩罚控制能量)
        # 增加 Q[0,0] 会让关节控制更刚硬，增加 R 会让动作更平缓省电
        Q = np.diag([1200.0, 15.0])  # [角度偏差惩罚, 角速度偏差惩罚]
        R = np.array([[0.01]])  # [控制力矩消耗惩罚]

        # 求解代数里卡提方程 (Continuous Algebraic Riccati Equation)
        P = solve_continuous_are(A, B, Q, R)

        # 计算最优反馈增益 K = R^(-1) B^T P = [Kp, Kd]
        self.K_lqr = np.dot(np.linalg.inv(R), np.dot(B.T, P)).flatten()
        self.Kp_lqr = self.K_lqr[0]  # 位置增益
        self.Kd_lqr = self.K_lqr[1]  # 速度增益

        print(f"⚡ LQR 最优反馈增益计算完成: Kp = {self.Kp_lqr:.2f}, Kd = {self.Kd_lqr:.2f}")

    def set_curriculum_stage(self, stage):
        self.curriculum_stage = stage
        print(f"🌍 环境已切换至课程学习阶段: Stage {stage}")

    def _get_obs(self):
        return np.concatenate([self.data.qpos.astype(np.float32), self.data.qvel.astype(np.float32)])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if self.curriculum_stage == 1:
            self.data.qpos[0] = 0.0
            self.data.qpos[1] = 0.5
        elif self.curriculum_stage == 2:
            self.data.qpos[0] = 0.0
            self.data.qpos[1] = 2.0
        else:
            self.data.qpos[0] = -0.3
            self.data.qpos[1] = 12.0

        self.data.qpos[2] = 0.42  # 初始站立高度
        self.data.qpos[3:7] = np.array([0.7071, 0.0, 0.0, 0.7071], dtype=np.float32)

        default_standing_angles = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float32)
        self.data.qpos[7:19] = default_standing_angles

        self.episode_steps = 0
        self.max_episode_steps = 1500
        self.last_y_position = self.data.qpos[1]
        self.stuck_steps = 0

        # 开局先用 LQR 控制稳定 20 个物理步，防止机器狗落地失重
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self.episode_steps += 1
        default_angles = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float32)

        # 1. 动作幅度稍微缩小一点，让初期探索更温和
        target_angles = default_angles + np.clip(action, -1.0, 1.0) * 0.2

        # 2. 获取当前关节角度与角速度
        current_angles = self.data.qpos[7:19]
        current_velocities = self.data.qvel[6:18]

        # 3. LQR 控制力矩计算
        control_torques = self.Kp_lqr * (target_angles - current_angles) - self.Kd_lqr * current_velocities

        # 4. 硬件力矩限幅
        torques_limits = np.array([23.7, 23.7, 45.43] * 4, dtype=np.float32)
        control_torques = np.clip(control_torques, -torques_limits, torques_limits)
        self.data.ctrl[:] = control_torques

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        base_pos = self.data.qpos[0:3]
        base_height = base_pos[2]

        # ==================== 🎯 优化的正向引导奖励 ====================
        # 1. 活着就有基础奖励
        reward = 2.0

        # 2. 高度奖励（平滑连续，而不是生硬的扣分）
        # 目标高度 0.42，高度越接近奖励越高 (最高可得 +5.0)
        height_error = abs(base_height - 0.42)
        height_reward = max(0.0, 5.0 * (1.0 - height_error * 3.0))
        reward += height_reward

        # 3. 轻轻的惩罚（防止动作过大或严重歪曲关节）
        posture_penalty = 0.001 * np.sum(np.square(current_angles - default_angles))
        action_penalty = 0.001 * np.sum(np.square(action))
        reward -= (posture_penalty + action_penalty)

        # 终止条件
        fallen = bool(base_height < 0.22)  # 高度低于 0.22 判为倒地
        terminated = fallen
        truncated = bool(self.episode_steps >= 1000)

        if fallen:
            reward -= 10.0  # 适度惩罚，避免总奖励直接崩塌成负几千

        return obs, reward, terminated, truncated, {}