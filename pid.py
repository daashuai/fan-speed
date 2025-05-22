import argparse
import os
import time
import numpy as np
from datetime import datetime
from cooling import CoolingEnv
from utils import (
    plot_speed_temp,
    calculate_energy,
    calculate_speed_smoothness,
    calculate_temp_deviation,
    calculate_max_change,
    calculate_temp_stabilization_time
)
from torch.utils.tensorboard import SummaryWriter
from pid_tuner import ZieglerNicholsAutoTuner

class BasicPID:
    def __init__(self, Kp, Ki, Kd, set_point, output_limits=(-1, 1)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = set_point
        self.output_min, self.output_max = output_limits
        
        # State variables
        self.last_error = 0
        self.integral = 0

    def compute(self, current_value):
        error = self.set_point - current_value
        
        # PID terms
        P = self.Kp * error
        self.integral += error
        I = self.Ki * self.integral
        D = self.Kd * (error - self.last_error)
        
        # Update state
        self.last_error = error
        
        # Clamp output
        output = P + I + D
        return np.clip(output, self.output_min, self.output_max)

# class PID:
#     def __init__(self, Kp, Ki, Kd, set_point, control_limits=(-1, 1), integral_limits=(-np.inf, np.inf), output_limits=(2000, 10000)):
#         self.Kp = Kp
#         self.Ki = Ki
#         self.Kd = Kd
#         self.set_point = set_point
#         self.output_min, self.output_max = output_limits
#         self.int_min, self.int_max = integral_limits
#         self.control_min, self.control_max = control_limits

#         self.last_error = 0
#         self.integral = 0
#         self.last_time = time.time()

#     def compute(self, current_value):
#         current_time = time.time()
#         # dt = current_time - self.last_time if self.last_time else 1.0
#         dt = 1

#         error = self.set_point - current_value

#         P = self.Kp * error
#         self.integral += error * dt
#         self.integral = np.clip(self.integral, self.int_min, self.int_max)
#         I = self.Ki * self.integral
#         D = self.Kd * (error - self.last_error) / dt

#         self.last_error = error
#         self.last_time = current_time

#         control = P + I + D
#         normalized_control = np.clip(control, self.control_min, self.control_max)
#         output = 2000 + (normalized_control + 1) * 4000
#         print(normalized_control)
#         return np.clip(output, self.output_min, self.output_max)


class PID:
    def __init__(self, Kp, Ki, Kd, set_point, output_limits=(-1, 1), integral_limits=(-np.inf, np.inf)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = set_point
        self.output_min, self.output_max = output_limits
        self.int_min, self.int_max = integral_limits

        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()

    def compute(self, current_value):
        current_time = time.time()
        # dt = current_time - self.last_time if self.last_time else 1.0
        dt = 1

        error = current_value - self.set_point

        P = self.Kp * error
        self.integral += error * dt
        self.integral = np.clip(self.integral, self.int_min, self.int_max)
        I = self.Ki * self.integral
        D = self.Kd * (error - self.last_error) / dt

        self.last_error = error
        self.last_time = current_time

        output = P + I + D
        return np.clip(output, self.output_min, self.output_max)


def pid(args):
    # 初始化实验目录和记录器
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join('exp_pid', time_stamp)
    os.makedirs(experiment_dir, exist_ok=True)
    writer = SummaryWriter(experiment_dir)
    
    # 创建环境
    env = CoolingEnv(
        obs_dim=args.obs_dim,
        workload_mode=args.workload_mode,
        mix_switch_interval=1000
    )
    
    # 初始化PID控制器
    pid = PID(
        Kp=args.Kp,
        Ki=args.Ki,
        Kd=args.Kd,
        set_point=env.temp_target,
        output_limits=(env.fan_min_speed, env.fan_max_speed)
    )

    # 训练循环（保持与SAC相同的epoch结构）
    best_score = -np.inf
    total_steps = args.epochs * args.steps_per_epoch
    
    for epoch in range(args.epochs):
        # 运行一个epoch
        state = env.reset()
        current_temp = state[-1]  # 获取平均温度
        episode_reward = 0
        steps = 0

        while steps < args.steps_per_epoch:
            # 生成PID控制信号
            fan_speed = pid.compute(current_temp)
            fan_speed_delta = fan_speed - env.fan_speed_current
            state, reward, done, _ = env.step(fan_speed_delta)
            # 记录数据
            episode_reward += reward
            current_temp = state[-1]
            steps += 1
            
            if done:
                break
        
        # 记录训练指标
        writer.add_scalar("Train/TotalReturn", episode_reward, epoch)
        plot_speed_temp(writer, epoch, env.speeds, env.temps)
        
        # 计算各项指标
        energy = calculate_energy(env.speeds)
        speed_smooth = calculate_speed_smoothness(env.speeds)
        temp_deviation = calculate_temp_deviation(env.temps)
        
        writer.add_scalar("Train/EnergyConsume", energy, epoch)
        writer.add_scalar("Train/SpeedSmooth", speed_smooth, epoch)
        writer.add_scalar("Train/TempDeviation", temp_deviation, epoch)
        
        # 测试评估（保持与SAC相同的评估逻辑）
        test_scores = []
        for _ in range(args.num_test_episodes):
            state = env.reset()
            current_temp = state[-1]
            total_reward = 0
            
            for _ in range(args.max_trajectory_len):
                fan_speed = pid.compute(current_temp)
                fan_speed_delta = fan_speed - env.fan_speed_current
                state, reward, done, _ = env.step(fan_speed_delta)
                total_reward += reward
                current_temp = state[-1]
                
                if done:
                    break
            
            test_scores.append(total_reward)
        
        avg_score = np.mean(test_scores)
        writer.add_scalar("Test/AverageReturn", avg_score, epoch)
        
        # 保存最佳参数
        if avg_score > best_score:
            best_score = avg_score
            best_params = {'Kp': args.Kp, 'Ki': args.Ki, 'Kd': args.Kd}
            np.save(os.path.join(experiment_dir, 'best_params.npy'), best_params)
            
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Return: {episode_reward:.1f} | "
              f"Test Score: {avg_score:.1f}")
    
    # 保存最终参数
    final_params = {'Kp': args.Kp, 'Ki': args.Ki, 'Kd': args.Kd}
    np.save(os.path.join(experiment_dir, 'final_params.npy'), final_params)
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Kp', type=float, default=100, help='比例增益（启用自动调参时作为初始值）')
    parser.add_argument('--Ki', type=float, default=0.5, help='积分增益（启用自动调参时作为初始值）')
    parser.add_argument('--Kd', type=float, default=30, help='微分增益（启用自动调参时作为初始值）')
    parser.add_argument('--obs_dim', type=int, default=5)
    parser.add_argument('--workload_mode', type=str, default='medium', 
                       choices=['low', 'medium', 'high', 'mixed'], 
                       help='工作负载模式')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=1000,
                       help='每个epoch的步数（每个PID控制周期）')
    parser.add_argument('--num_test_episodes', type=int, default=5)
    parser.add_argument('--max_trajectory_len', type=int, default=1000)
    parser.add_argument('--auto_tune', action='store_true',
                       help='启用Ziegler-Nichols自动调参')
    parser.add_argument('--tune_steps', type=int, default=500,
                       help='自动调参阶段的步数')
    args = parser.parse_args()

    # 自动调参阶段（独立环境）
    if args.auto_tune:
        print("\n=== 启动自动调参流程 ===")
        tune_env = CoolingEnv(
            obs_dim=args.obs_dim,
            workload_mode=args.workload_mode,
            mix_switch_interval=1000
        )
        
        tuner = ZieglerNicholsAutoTuner(
            tune_env,
            max_kp=50.0,
            tolerance=0.5,
            max_attempts=30
        )
        
        try:
            tuned_params = tuner.tune()
            if tuned_params:
                print(f"调参成功！参数更新: Kp={tuned_params['Kp']:.2f}, "
                      f"Ki={tuned_params['Ki']:.2f}, Kd={tuned_params['Kd']:.2f}")
                
                # 覆盖默认参数
                args.Kp = tuned_params['Kp']
                args.Ki = tuned_params['Ki']
                args.Kd = tuned_params['Kd']
                
                # 保存调参过程数据
                time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                np.savez(f'tuning_data_{time_stamp}.npz',
                        temps=tuner.oscillation_data,
                        speeds=tune_env.speeds,
                        params=tuned_params)
        except Exception as e:
            print(f"自动调参失败: {str(e)}，使用默认参数")
            args.auto_tune = False

    # 主实验执行（使用调参后或默认参数）
    print("\n=== 开始主实验 ===")
    print(f"最终参数: Kp={args.Kp:.2f}, Ki={args.Ki:.2f}, Kd={args.Kd:.2f}")
    pid(args)
