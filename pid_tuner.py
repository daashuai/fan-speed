import numpy as np
from cooling import CoolingEnv

class ZieglerNicholsAutoTuner:
    def __init__(self, env, max_kp=100.0, tolerance=0.1, max_attempts=50):
        """
        :param env: 冷却控制环境实例
        :param max_kp: 最大允许的比例增益（安全限制）
        :param tolerance: 振荡幅度稳定判定阈值
        :param max_attempts: 最大尝试次数（防止无限循环）
        """
        self.env = env
        self.max_kp = max_kp
        self.tolerance = tolerance
        self.max_attempts = max_attempts
        self.oscillation_data = []

    def detect_oscillation(self, temps, min_cycles=4):
        """检测温度序列是否呈现稳定等幅振荡"""
        peaks = []
        valleys = []
        
        # 寻找波峰波谷
        for i in range(1, len(temps)-1):
            if temps[i] > temps[i-1] and temps[i] > temps[i+1]:
                peaks.append(temps[i])
            elif temps[i] < temps[i-1] and temps[i] < temps[i+1]:
                valleys.append(temps[i])
        
        # 需要至少4个完整周期
        if len(peaks) < min_cycles or len(valleys) < min_cycles:
            return False, 0, 0
        
        # 计算振幅稳定性
        peak_std = np.std(peaks[-min_cycles:])
        valley_std = np.std(valleys[-min_cycles:])
        amplitude_stable = (peak_std < self.tolerance) and (valley_std < self.tolerance)
        
        # 计算平均周期
        time_steps = len(temps) // (min_cycles * 2)
        return amplitude_stable, np.mean([p-v for p,v in zip(peaks, valleys)]), time_steps

    def find_critical_point(self, steps_per_attempt=200):
        """寻找临界振荡点"""
        kp = 0.1
        delta_kp = 0.5  # Kp增量步长
        
        for attempt in range(self.max_attempts):
            # 重置环境并设置纯比例控制
            state = self.env.reset()
            self.env.speed = self.env.fan_min_speed  # 重置风扇速度
            current_temp = state[-1]
            temps = []
            
            # 运行仿真
            for _ in range(steps_per_attempt):
                error = self.env.temp_target - current_temp
                action = np.clip(kp * error, -1, 1)  # 动作空间限制
                state, _, done, _ = self.env.step(np.array([action]))
                current_temp = state[-1]
                temps.append(current_temp)
                if done: break
            
            # 检测振荡
            stable, amplitude, period = self.detect_oscillation(temps)
            
            if stable:
                # 计算临界参数
                Ku = kp
                Tu = period * self.env.interval  # 转换为秒
                return Ku, Tu, temps
            else:
                # 安全保护：振幅超过安全阈值时终止
                if np.ptp(temps) > 50:  
                    raise RuntimeError("系统振荡幅度过大，可能损坏设备")
                
                # 动态调整Kp增量
                kp += delta_kp * (1.2 if attempt < 10 else 1.0)
                if kp > self.max_kp:
                    raise RuntimeError(f"达到最大Kp限制{self.max_kp}仍未找到临界点")
        
        raise RuntimeError("超过最大尝试次数仍未找到临界点")

    def tune(self):
        """执行完整的Ziegler-Nichols参数整定"""
        try:
            Ku, Tu, oscillation_data = self.find_critical_point()
            self.oscillation_data = oscillation_data
            
            # 按ZN公式计算PID参数
            Kp = 0.6 * Ku
            Ki = 2 * Kp / Tu
            Kd = Kp * Tu / 8
            
            return {
                'Kp': round(Kp, 3),
                'Ki': round(Ki, 3),
                'Kd': round(Kd, 3),
                'Ku': round(Ku, 3),
                'Tu': round(Tu, 3)
            }
        except Exception as e:
            print(f"自动调参失败: {str(e)}")
            return None
