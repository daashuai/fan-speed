# coding: utf-8
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import math
import random
import gym
from gym import spaces

### === 第一步：加载并处理数据 === ###

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # 去除列名空格
    core_cols = [col for col in df.columns if col.startswith("Core")]
    df["Temp_Avg"] = df[core_cols].mean(axis=1)
    fan_speeds = df["Fan1_Speed"].values
    T_real = df["Temp_Avg"].values
    task_levels = df["task_level"].values if "task_level" in df.columns else np.ones(len(df)) * 2.4
    return fan_speeds, task_levels, T_real

### === 第二步：模拟函数和损失函数 === ###

def simulate_temp(fan_speeds, task_levels, params, T0, temp_ambient=20.0):
    heat_add, heat_remove, fan_effect_modifer, heat_on_temp = params
    T = [T0]
    heat = (T0 - temp_ambient) * heat_on_temp
    for speed, task_level in zip(fan_speeds, task_levels):
        delta_heat = heat_add * task_level - heat_remove * (speed / fan_effect_modifer)
        heat += delta_heat
        T_new = temp_ambient + heat / heat_on_temp
        T.append(T_new)
    return np.array(T[:-1])

def loss(params, fan_speeds, task_levels, T_real, T0):
    T_sim = simulate_temp(fan_speeds, task_levels, params, T0)
    return np.mean((T_sim - T_real) ** 2)

### === 第三步：拟合参数 === ###

def fit_model(fan_speeds, task_levels, T_real):
    initial_guess = [500, 300, 1000, 1000]
    result = minimize(loss, initial_guess, args=(fan_speeds, task_levels, T_real, T_real[0]), method='Nelder-Mead')
    return result.x

### === 第四步：定义带注入参数的环境 === ###

class CoolingEnv(gym.Env):
    def __init__(self, heat_add, heat_remove, fan_effect_modifer, heat_on_temp):
        super().__init__()
        self.heat_base = 2000
        self.heat_add  = heat_add
        self.heat_on_temp = heat_on_temp
        self.heat_remove  = heat_remove
        self.heat_modifiers = [1.0, 1.5, 2.0, 2.4, 3.0]  # 示例：不同负载等级
        self.fan_effect_modifer = fan_effect_modifer
        self.noise = 0.2

        self.fan_max_speed = 10000
        self.fan_min_speed = 2000
        self.fan_speed_add_high = 200
        self.fan_speed_add_low = -200

        self.temp_target = 40
        self.temp_ambient = 20

        self.max_steps = 1000
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
        self.observation_space = spaces.Discrete(6)
        self.reset()

    def step(self, action):
        speed_add = int(action * self.fan_speed_add_high)
        self.speed = np.clip(self.speed + speed_add, self.fan_min_speed, self.fan_max_speed)

        task_level = random.choice(self.heat_modifiers)
        heat_increase = self.heat_add * task_level - self.heat_remove * (self.speed / self.fan_effect_modifer)
        self.heat_gross += heat_increase

        self.temp = self.temp_ambient + self.heat_gross / self.heat_on_temp + math.sin(self.steps/1000) * self.noise
        self.temps.append(self.temp)
        self.speeds.append(self.speed)

        if self.temp > self.temp_target + 2:
            r1 = -math.log(self.temp - self.temp_target -1)
        elif self.temp >= self.temp_target - 2 and self.temp <= self.temp_target + 2:
            r1 = 2
        elif self.temp < self.temp_target - 2 and self.temp >= self.temp_ambient:
            r1 = 1 / (1 + (self.temp_target - self.temp - 2))
        elif self.temp < self.temp_ambient:
            r1 = -math.log(self.temp_ambient - self.temp + 1)

        speed_prev = self.speeds[-2] if len(self.speeds) >= 2 else self.fan_min_speed
        r3 = -math.log(abs(speed_add) + 1)

        reward = r1 + r3

        self.state.pop(0)
        self.state.append(round(self.temp, 2))
        mean = sum(self.state)/len(self.state)
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
    fan_speeds, task_levels, T_real = load_and_process_data("4core_120s.csv")
    params = fit_model(fan_speeds, task_levels, T_real)
    print("Fitted parameters:", params)

    env = CoolingEnv(*params)
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

    print("Final temps:", env.temps)
