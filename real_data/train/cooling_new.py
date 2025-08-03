import pandas as pd
import numpy as np
from scipy.optimize import minimize
import math
import random
import gym
from gym import spaces

pd.set_option('display.float_format', '{:.2f}'.format)
np.set_printoptions(suppress=True)

### === 第一步：加载数据 === ###

def _load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    core_cols = [col for col in df.columns if col.startswith("Temp")]
    df["Temp_Avg"] = df[core_cols].mean(axis=1)
    fan_speeds = df["Fan(RPM)"].values
    T_real = df["Temp_Avg"].values
    task_levels = df["task_level"].values if "Power(W)" in df.columns else np.ones(len(df)) * 2.4
    return fan_speeds, task_levels, T_real

def load_and_process_data(file_path):
    import pandas as pd
    import numpy as np

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # 计算温度均值列
    core_cols = [col for col in df.columns if col.startswith("Temp")]
    df["Temp_Avg"] = df[core_cols].mean(axis=1)

    # 去掉前30条数据
    df = df.iloc[30:].reset_index(drop=True)

    # 计算每10条数据的均值
    def grouped_mean(arr):
        n = len(arr)
        n_groups = n // 10
        arr = arr[:n_groups * 10].reshape(n_groups, 10)
        return arr.mean(axis=1)

    fan_speeds = grouped_mean(df["Fan(RPM)"].values)
    T_real = grouped_mean(df["Temp_Avg"].values)
    
    if "Power(W)" in df.columns:
        task_levels = grouped_mean(df["Power(W)"].values)
    else:
        task_levels = np.ones_like(fan_speeds) * 2.4

    return fan_speeds, task_levels, T_real


### === 第二步：模拟温度和损失函数 === ###

def simulate_temp(fan_speeds, task_levels, params, T0, temp_ambient=20.0):
    heat_add, fan_efficiency, heat_on_temp = params
    T = [T0]
    heat = (T0 - temp_ambient) * heat_on_temp
    for speed, task_level in zip(fan_speeds, task_levels):
        delta_heat = heat_add * task_level - fan_efficiency * speed
        heat += delta_heat
        T_new = temp_ambient + heat / heat_on_temp
        T.append(T_new)
    return np.array(T[:-1])

def loss(params, fan_speeds, task_levels, T_real, T0):
    T_sim = simulate_temp(fan_speeds, task_levels, params, T0)
    return np.mean((T_sim - T_real) ** 2)

### === 第三步：拟合参数 === ###

def fit_model(fan_speeds, task_levels, T_real):
    initial_guess = [500, 0.1, 10000]  # heat_add, fan_efficiency, heat_on_temp
    bounds = [
        (1, 50000),       # heat_add
        (1, 10),       # fan_efficiency
        (10, 50000)       # heat_on_temp
    ]
    result = minimize(
        loss,
        initial_guess,
        args=(fan_speeds, task_levels, T_real, T_real[0]),
        method='L-BFGS-B',
        bounds=bounds
    )
    return result.x

### === 第四步：构造 Gym 环境 === ###

class CoolingEnv(gym.Env):
    def __init__(self, heat_add, fan_efficiency, heat_on_temp):
        super().__init__()
        self.heat_base = 2000
        self.heat_add = heat_add
        self.fan_efficiency = fan_efficiency
        self.heat_on_temp = heat_on_temp
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

        task_level = random.choice(self.heat_modifiers)
        heat_increase = self.heat_add * task_level - self.fan_efficiency * self.speed
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

### === 主程序入口 === ###

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

