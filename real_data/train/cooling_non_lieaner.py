import pandas as pd
import numpy as np
from scipy.optimize import minimize
import math
import random
import gym
from gym import spaces

# 禁用科学计数法
pd.set_option('display.float_format', '{:.2f}'.format)
np.set_printoptions(suppress=True)

### === 1. 数据处理（加入归一化） === ###
def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    core_cols = [col for col in df.columns if col.startswith("Temp")]
    df["Temp_Avg"] = df[core_cols].mean(axis=1)

    fan_speeds = df["Fan(RPM)"].values / 4000 # 归一化
    task_levels = df["Power(W)"].values / 85 if "Power(W)" in df.columns else np.ones(len(df)) * (2.4 / 3.0)
    T_real = df["Temp_Avg"].values
    return fan_speeds, task_levels, T_real

### === 2. 模拟和损失函数 === ###
# def simulate_temp(fan_speeds, task_levels, params, T0, temp_ambient=50.0):
#     heat_add, fan_efficiency, alpha, heat_on_temp = params
#     # alpha=0.1
#     T = [T0]
#     heat = (T0 - temp_ambient) * heat_on_temp
#     for speed, task_level in zip(fan_speeds, task_levels):
#         cooling = fan_efficiency * (1 - np.exp(-alpha * speed))  # 注意speed已经归一化
#         delta_heat = heat_add * task_level - cooling
#         heat += delta_heat
#         T_new = temp_ambient + heat / heat_on_temp
#         T.append(T_new)
#     return np.array(T[:-1])
def simulate_temp(fan_speeds, task_levels, params, T0, temp_ambient=50.0):
    heat_add, fan_efficiency, alpha, beta, heat_on_temp, heat_saturation = params
    T = [T0]
    heat = (T0 - temp_ambient) * heat_on_temp
    
    for speed, task_level in zip(fan_speeds, task_levels):
        current_temp = temp_ambient + heat / heat_on_temp
        
        # 动态热生成：高温时效率降低（模拟热饱和效应）
        heat_gen = heat_add * task_level / (1 + heat_saturation * (current_temp - temp_ambient))
        
        # 非线性散热
        cooling = fan_efficiency * (1 - np.exp(-alpha * speed - beta * speed**2))
        
        delta_heat = heat_gen - cooling
        heat += delta_heat
        T_new = temp_ambient + heat / heat_on_temp
        T.append(T_new)
    
    return np.array(T[:-1])

def loss(params, fan_speeds, task_levels, T_real, T0):
    T_sim = simulate_temp(fan_speeds, task_levels, params, T0)
    return np.mean((T_sim - T_real) ** 2)

### === 3. 拟合参数 === ###
def fit_model(fan_speeds, task_levels, T_real):
    initial_guess = [1000, 3000, 5, 5, 1000, 1000]
    bounds = [
        (1, None),    # heat_add
        (10, None),   # fan_efficiency
        (0.1, None),    # alpha（归一化后可放宽范围）
        (0.1, None), 
        (1, None), 
        (1, None)    # heat_on_temp
    ]
    result = minimize(
        loss,
        initial_guess,
        args=(fan_speeds, task_levels, T_real, T_real[0]),
        method='L-BFGS-B',
        bounds=bounds
    )
    return result.x

### === 4. 环境定义 === ###
class CoolingEnv(gym.Env):
    def __init__(self, heat_add, fan_efficiency, alpha, heat_on_temp):
        super().__init__()
        self.heat_add = heat_add
        self.fan_efficiency = fan_efficiency
        self.alpha = alpha
        self.heat_on_temp = heat_on_temp

        self.heat_base = 2000
        self.temp_ambient = 20
        self.temp_target = 40
        self.noise = 0.2
        self.max_steps = 1000

        self.heat_modifiers = [1.0, 1.5, 2.0, 2.4, 3.0]
        self.fan_max_speed = 10000
        self.fan_min_speed = 2000
        self.fan_speed_add_high = 200

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
        self.observation_space = spaces.Discrete(6)
        self.reset()

    def step(self, action):
        speed_add = int(action * self.fan_speed_add_high)
        self.speed = np.clip(self.speed + speed_add, self.fan_min_speed, self.fan_max_speed)

        task_level = random.choice(self.heat_modifiers) / 3.0  # 归一化
        speed_norm = self.speed / 10000.0  # 归一化
        cooling = self.fan_efficiency * (1 - np.exp(-self.alpha * speed_norm))

        heat_increase = self.heat_add * task_level - cooling
        self.heat_gross += heat_increase

        self.temp = self.temp_ambient + self.heat_gross / self.heat_on_temp + math.sin(self.steps / 1000) * self.noise
        self.temps.append(self.temp)
        self.speeds.append(self.speed)

        if self.temp > self.temp_target + 2:
            r1 = -math.log(self.temp - self.temp_target - 1)
        elif self.temp >= self.temp_target - 2 and self.temp <= self.temp_target + 2:
            r1 = 2
        elif self.temp < self.temp_target - 2 and self.temp >= self.temp_ambient:
            r1 = 1 / (1 + (self.temp_target - self.temp - 2))
        else:
            r1 = -math.log(self.temp_ambient - self.temp + 1)

        r3 = -math.log(abs(speed_add) + 1)
        reward = r1 + r3

        self.state.pop(0)
        self.state.append(round(self.temp, 2))
        mean = sum(self.state) / len(self.state)
        self.state_normal = [x - mean for x in self.state] + [mean]

        self.steps += 1
        done = self.steps >= self.max_steps
        return self.state_normal, reward, done, {}

    def reset(self):
        self.speed = self.fan_min_speed
        self.heat_gross = self.heat_base
        self.temp = self.temp_ambient
        self.steps = 0
        self.state = [self.temp_ambient] * 5
        mean = self.temp_ambient
        self.state_normal = [0] * 5 + [mean]
        self.temps = []
        self.speeds = []
        return self.state_normal

if __name__ == "__main__":
    data_path = [
        "cpu_0%.csv",
        "cpu_10%.csv",
        "cpu_20%.csv",
        "cpu_30%.csv",
        "cpu_40%.csv",
        "cpu_50%.csv",
        "cpu_60%.csv",
        "cpu_70%.csv",
        "cpu_80%.csv",
        "cpu_90%.csv",
        "cpu_100%.csv",
        "all.csv"
    ]

    for path in data_path:
        fan_speeds, task_levels, T_real = load_and_process_data(path)
        params = fit_model(fan_speeds, task_levels, T_real)
        print(path)
        print("Fitted parameters:", params)

        # env = CoolingEnv(*params)
        # obs = env.reset()
        # done = False
        # while not done:
        #     action = env.action_space.sample()
        #     obs, reward, done, _ = env.step(action)

        # print("Final temps:", env.temps)

