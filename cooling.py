import sys
import time
import math
import random
# import logger
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding

class CoolingEnv(gym.Env):
    """
    Description:

    Source:
        This environment corresponds to the version of the cpu cooling system
        described by ludashuai

    Observation: 
        Type: Discrete(6)
        Num	Observation                
        0   History Temperature	minus Mean 
        1	History Temperature	minus Mean 
        2	History Temperature	minus Mean   
        3	History Temperature	minus Mean                      
        4	History Temperature	minus Mean                             
        5   Mean of History Temperature
        
    Actions:
        Type: Discrete(2)
        Num	Action          Min         Max
        0   Fan Speed       2000	    10000
        

    Reward:
        Reward is current temperature minus target temperature for every step taken,
        including the termination step

    Starting State:

    Episode Termination:
        Considered solved when the total step bigger than 1000.
    """

    def __init__(self, obs_dim=5, workload_mode='medium', mix_switch_interval=200):
        self.heat_base = 2000
        self.heat_add  = 500
        self.heat_on_temp = 1000
        self.heat_remove  = 300
        self.heat_modifiers = [2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4]
        ## 定义不同负载模式的发热系数
        self.workload_config = {
            'low': [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4],
            'medium': [2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4],
            'high': [3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4, 3.4]
        } 
        self.modefier_gap = 1000
    
        # 新增工作模式配置
        self.workload_mode = workload_mode.lower()
        self.mix_switch_interval = mix_switch_interval
        self.step_counter = 0
        self.workload_mode_mix = 'low'

        self.fan_effect_modifer = 1000
        self.noise = 0.2
        self.fan_accuracy = 10
        self.fan_max_speed = 10000
        self.fan_min_speed = 2000
        self.fan_speed_add_high = 200
        self.fan_speed_add_low = -200
        self.fan_speed_current = self.fan_min_speed

        self.temp_target = 40
        self.temp_ambient = 20

        self.interval = 10
        
        self.max_steps = 1000
        

        self.obs_dim = obs_dim
        self.action_space = spaces.Box(low=-1,
                high=1, shape=(1,), dtype=float)
        self.observation_space = spaces.Discrete(self.obs_dim + 1)

        self.speed = self.fan_min_speed
        self.steps = 0
        self.steps_total = 0
        self.temp = self.temp_ambient
        self.temps = []
        self.speeds = []
        self.counter = 0

        # self.state = [self.temp_ambient, self.temp_ambient, self.temp_ambient, self.temp_ambient, self.temp_ambient]
        self.state = [self.temp_ambient] * self.obs_dim
        mean = sum(self.state)/len(self.state)
        self.state_normal = []
        for x in self.state:
            self.state_normal.append(x - mean)
        self.state_normal.append(mean)

    def step(self, action):

        fan_speed_delta = action
        self.fan_speed_current += fan_speed_delta

        # 混合模式切换逻辑
        if self.workload_mode == 'mixed':
            self.step_counter += 1
            if self.step_counter >= self.mix_switch_interval:
                self.step_counter = 0
                # 循环切换子模式：low->medium->high->low
                if self.workload_mode_mix == 'low':
                    self.workload_mode_mix = 'medium'
                elif self.workload_mode_mix == 'medium':
                    self.workload_mode_mix = 'high'
                else:
                    self.workload_mode_mix = 'low'

        if self.workload_mode == 'mixed':
            self.heat_modifiers = self.workload_config[self.workload_mode_mix]
        else:
            self.heat_modifiers = self.workload_config[self.workload_mode]

        s = self.temp

        # action = self.action_space[action]

        # self.speed_add = int((action - self.action_space.low) *
        #         (self.fan_speed_add_high - self.fan_speed_add_low) +
        #         self.fan_speed_add_low)

        # self.speed = int((action-self.action_space.low)*(self.fan_max_speed - self.fan_min_speed) +
        #         self.fan_min_speed)
        
        speed_add  = int(action* self.fan_speed_add_high)
        self.speed += speed_add
        
        if self.speed > self.fan_max_speed:
            self.speed = self.fan_max_speed 
        if self.speed < self.fan_min_speed:
            self.speed = self.fan_min_speed 
        
        # 加入阈值之后, 不收敛, 原因未知
        # # to smooth the curve of fan speed
        # if len(self.speeds) >= 1:
        #     speed_prev = self.speeds[len(self.speeds) - 1]
        #     if self.speed - speed_prev < 10:
        #         self.speed = speed_prev 
         
        # task_level = HEAT_MODIFIERS[int(self.steps/MODEFIER_GAP)]
        task_level = self.heat_modifiers[random.randint(0, len(self.heat_modifiers)-1)]

        heat_increase = self.heat_add* task_level - self.heat_remove*(self.speed/self.fan_effect_modifer)

        self.heat_gross += heat_increase

        self.temp = self.temp_ambient + self.heat_gross/self.heat_on_temp + math.sin(self.steps/1000) * self.noise 
        
        self.speeds.append(self.speed)
        self.temps.append(self.temp)

        # reward function
        # target的偏移量小
        # 风扇转速尽量小
        # 风扇转速与温度上升尽量呈现正相关性

        # if self.temp > TEMP_TARGET:
        #     reward = TEMP_TARGET - self.temp
        # elif self.temp <= TEMP_TARGET:
        #     reward = 1




        if self.steps == self.max_steps - 1:
            done = True
        else:
            done = False
            self.steps += 1

        self.steps_total += 1
#         if abs(self.temp - self.temp_target) < 2:
#             self.step_counter += 1
#         else:
#             self.step_couner = 0
# 
#         if self.step_counter > 10:
#             # done = True
#             reward = 10
#         else:
#             reward = -abs(self.temp - self.temp_ambient)

        # 如果温度大于self.temp_target 就给与一个惩罚
        # 将温度分为三个档, 在不同档位的温度, 给予不同的奖励和惩罚
        # reward = 0
        # if self.temp > self.temp_target:
        #     r1 = -math.log(self.temp-self.temp_target + 1)
        # elif self.temp >= self.temp_ambient and self.temp <= self.temp_target: 
        #     r1 = 1/(1 + (self.temp_target -self.temp))
        # elif self.temp < self.temp_ambient:
        #     r1 = -math.log(self.temp_ambient - self.temp + 1)
        
        # 将温度分为四个区间
        reward = 0
        if self.temp > self.temp_target + 2:
            r1 = -math.log(self.temp - self.temp_target -1 )
        elif self.temp >= self.temp_target - 2 and self.temp <= self.temp_target + 2:
            r1 = 2
        elif self.temp < self.temp_target - 2  and self.temp >= self.temp_ambient:
            r1 = 1/(1 + (self.temp_target - self.temp - 2))
        elif self.temp < self.temp_ambient:
            r1 = -math.log(self.temp_ambient -self.temp + 1)

        # # 如果风扇转速过快, 就给予一个惩罚
        # # 5500 之下的转速给予奖励, 之上的转速给予惩罚
        # r2 = -0.0002 * (self.speed - self.fan_min_speed - 3500) 

        # # 如果风扇转速变化过于激烈, 给予惩罚.
        # epoch = (self.steps_total +1) // 6000 

        # if  epoch <= 20:
        #     c = 0.0004 + 0.0028*epoch/20
        # else:
        #     c = 0.0032


        # speed_prev = [self.speed, self.speed, self.speed, self.speed, self.speed]
        # if len(self.speeds) > 6:
        #     speed_prev[0] = self.speeds[len(self.speeds) - 2]
        #     speed_prev[1] = self.speeds[len(self.speeds) - 3]
        #     speed_prev[2] = self.speeds[len(self.speeds) - 4]
        #     speed_prev[3] = self.speeds[len(self.speeds) - 5]
        #     speed_prev[4] = self.speeds[len(self.speeds) - 6]


        # # total = abs(self.speed - speed_prev[0]) + abs(self.speed - speed_prev[1]) + \
        # # abs(self.speed - speed_prev[2]) + abs(self.speed - speed_prev[3]) + abs(self.speed - speed_prev[4])
        # 
        # 
        # # r3 = -c * total / 5
        # r3 = -c * abs(self.speed - speed_prev[0]) 

        if len(self.speeds) >= 2:
            speed_prev = self.speeds[len(self.speeds) - 2]
        else:
            speed_prev = self.fan_min_speed

        # r3 = -0.004 * abs(self.speed- speed_prev)
        r3 = -math.log(abs(speed_add) + 1)

        # # 设计一个风扇速度波动积分

        # # 设计一个完成奖励
        # # 如果温度能够在三十步里保持稳定, 且风扇转速变化累计小于一个阈值, 则视为完成
        # # r4 = 
        # r4 = 0
        # if abs(self.temp - self.temp_target) < 2:
        #     self.counter += 1
        # else:
        #     self.counter = 0
        # 
        # total = 0
        # r4 = 0
        # if len(self.speeds) > 30:
        #     for i in range(30):
        #         total += abs(self.speed - self.speeds[len(self.speeds) - i - 2])

        #     if self.counter > 30 and total < 250:
        #         r4 = 5
        #     if self.counter > 30 and total < 150:
        #         r4 = 10
        #     if self.counter > 30 and total < 100:
        #         r4 = 15
        #     if self.counter > 30 and total < 20:
        #         r4 = 100

        # # 设计一个终止条件
        # # r5 = 
        # reward = r1 + r2 + r3 + r4
        reward = r1 + r3

        # 采用指数形式的奖励
        # 采用平方差形式的奖励
        # reward = -abs(self.temp - self.temp_target)
        # reward = -math.exp(abs(self.temp - self.temp_target))
        # reward = -(self.temp - self.temp_target) ** 2
        # reward = -math.log(abs(self.temp -self.temp_target) + 1) 
        # s_ = "%.1f"%self.temp
        # s_ = round(self.temp, 1)
        # s_ = int(self.temp)
        self.state.pop(0)
        self.state.append(round(self.temp, 2))
        mean = sum(self.state)/len(self.state)
        self.state_normal = []
        for x in self.state:
            self.state_normal.append(x - mean)
        self.state_normal.append(mean)
        s_ = self.state_normal
        
        return s_, reward, done, {}

    def reset(self):
        self.speed = self.fan_min_speed 
        self.heat_gross = self.heat_base 
        self.temp = self.temp_ambient 
        self.steps = 0
        self.step_counter = 0

        # self.state = [self.temp_ambient, self.temp_ambient, self.temp_ambient, self.temp_ambient, self.temp_ambient]
        self.state = [self.temp_ambient] * self.obs_dim
        mean = sum(self.state)/len(self.state)
        self.state_normal = []
        for x in self.state:
            self.state_normal.append(x - mean)
        self.state_normal.append(mean)

        self.temps = []
        self.speeds = []

        return self.state_normal

    def plot(self, speeds, temps):
        view_start = 100
        speed_max = max(speeds)
        speed_min = min(speeds[view_start:])
        temp_max = max(temps)
        temp_min = min(temps[view_start:])
        test_length = self.max_steps 
        
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(15,8))
        ax2 = ax1.twinx()
        ax1.plot(range(0, test_length), speeds, 'green', linewidth=3, label='Fan Speed')
        ax2.plot(range(0, test_length), temps, 'red', linewidth=3, label='Temperature')
        ax1.set_ylabel("Fan Speed")
        ax2.set_ylabel("Temperature")

        ax1.annotate(str(speed_max), xy=(speeds.index(speed_max), speed_max),
             xytext=(speeds.index(speed_max) + 100, speed_max + 350), arrowprops=dict(facecolor='black', shrink=1))
        ax1.annotate(str(speed_min), xy=(speeds.index(speed_min), speed_min),
             xytext=(speeds.index(speed_min) + 100, speed_min - 500), arrowprops=dict(facecolor='black', shrink=1))

        ax2.annotate(str(temp_max), xy=(temps.index(temp_max), temp_max),
             xytext=(temps.index(temp_max) - 100, temp_max + 2), arrowprops=dict(facecolor='orange', shrink=1))
        ax2.annotate(str(temp_min), xy=(temps.index(temp_min), temp_min),
             xytext=(temps.index(temp_min) + 50, temp_min - 2), arrowprops=dict(facecolor='orange', shrink=1))
        fig.legend(loc='upper right')

        fig.savefig('./image/' + 'sac-' + time.strftime('%Y-%m-%d %H:%M', time.localtime()) + '.jpg')

        terminal = sys.stdout
        sys.stdout = open('./image/speed-temp.log','a')
        print("speeds:")
        print(speeds)
        print("temps:")
        print(temps)
        sys.stdout = terminal

if __name__ == "__main__":
   env = CoolingEnv()
   obs_dim = env.observation_space.n
   act_dim = env.action_space.shape[0]
   act_limit = env.action_space.high[0]

   env1 = gym.make('HalfCheetah-v2')
   obs_dim1 = env1.observation_space.shape
   act_dim1 = env1.action_space.shape[0]
   act_limit1 = env1.action_space.high[0]
