import matplotlib.pyplot as plt
import numpy as np


def plot_robot_kinematics():
    # 示例：模拟仿真过程中的时间步与关节数据
    time_steps = np.linspace(0, 10, 500)

    # 模拟髋关节与脊椎关节的角度变化
    hip_angles = np.sin(time_steps * 3) * 0.5
    spine_angles = np.cos(time_steps * 1.5) * 0.2

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_steps, hip_angles, label='FR_hip Joint Angle (rad)', color='green')
    plt.plot(time_steps, spine_angles, label='Spine Joint Angle (rad)', color='orange')
    plt.title('Joint Trajectories during Locomotion')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid(True)

    # 模拟质心高度轨迹
    com_height = 0.3 + np.sin(time_steps * 6) * 0.02
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, com_height, label='Center of Mass (CoM) Height', color='purple')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_robot_kinematics()