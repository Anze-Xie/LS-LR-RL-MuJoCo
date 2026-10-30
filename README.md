# LS-LR-RL-MuJoCo
Locomotion Simulation of Legged Robot Based on Reinforcement Learning in MuJoCo

### 推荐的项目目录结构

```text
├── assets/                    # 机器人模型与复杂地形描述文件 (MJCF / XML / URDF)
│   ├── robots/                # 机器人模型（如 Unitree A1 / Go1 / 人形等）
│   │   └── robot.xml
│   └── terrains/              # 复杂环境模型（平地、斜坡、楼梯、梅花桩等）
│       ├── flat.xml
│       ├── slope.xml
│       ├── stairs.xml
│       └── stepping_stones.xml
│
├── envs/                      # Gym / Gymnasium 标准强化学习环境封装
│   ├── __init__.py
│   ├── base_env.py            # MuJoCo 仿真基础环境（状态/动作空间、观测值构建）
│   ├── legged_robot_env.py    # 腿足机器人核心动力学逻辑与奖励函数 (Reward Formulation)
│   └── terrain_wrapper.py     # 复杂地形加载与状态切换包装器
│
├── algos/                     # 强化学习算法实现（如 PPO, SAC 等）
│   ├── __init__.py
│   ├── ppo.py                 # PPO 策略训练核心算法
│   ├── actor_critic.py        # 网络结构定义 (Actor/Critic)
│   └── utils.py               # Buffer、GVE、优势估计等工具函数
│
├── pretrain/                # 训练好的模型权重 (注意：不要把超大 checkpoint 提上去)
│   ├── best_model_flat.pth    # 平地/斜坡策略权重
│   └── best_model_stairs.pth  # 楼梯/梅花桩策略权重
│
├── scripts/                   # 运行脚本
│   ├── train.py               # 启动训练入口脚本
│   ├── eval.py                # 策略评估与渲染演示脚本（用于展示运动能力）
│   └── export_onnx.py         # 策略导出工具（可选）
│
├── analysis/                  # 训练过程与运动动力学分析脚本（对应作业要求(4)）
│   ├── plot_rewards.py        # 训练收敛曲线绘制
│   └── plot_kinematics.py     # 关节角/速度/质心轨迹分析
│
├── docs/                      # 汇报 PPT 截图或实验分析报告材料
│   └── README_media/          # 存放演示 GIF 或截图
│
├── .gitignore                 # Git 忽略规则文件
├── README.md                  # 项目主文档（包含项目简介、安装方法、运行命令）
└── requirements.txt           # 项目依赖包清单