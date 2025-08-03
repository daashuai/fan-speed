import pandas as pd
import numpy as np
from scipy.optimize import minimize

pd.set_option('display.float_format', '{:.2f}'.format)
np.set_printoptions(suppress=True)

def load_and_process_data(file_path, max_fan_rpm=4000, max_power_w=85):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    core_cols = [col for col in df.columns if col.startswith("Temp")]
    df["Temp_Avg"] = df[core_cols].mean(axis=1)
    df = df[(df["Temp_Avg"] > 30) & (df["Temp_Avg"] < 120)]  
    
    fan_speeds = np.clip(df["Fan(RPM)"].values, 0, max_fan_rpm) / max_fan_rpm
    
    if "Power(W)" in df.columns:
        task_levels = np.clip(df["Power(W)"].values, 0, max_power_w) / max_power_w
    else:
        task_levels = np.ones(len(df)) * 0.8  
        
    T_real = df["Temp_Avg"].values
    return fan_speeds, task_levels, T_real

import numpy as np

def simulate_temp(fan_speeds, task_levels, params, T0, temp_ambient=50.0):
    (heat_add_base, fan_efficiency, alpha, beta, 
     heat_on_temp_base, power_threshold, temp_target) = params
    
    # 散热分配比例（风扇vs水箱）
    fan_eff_ratio = 0.75  # 风扇主导散热
    tank_cooling = fan_efficiency * (1 - fan_eff_ratio)
    
    T = [T0]
    heat = (T0 - temp_ambient) * heat_on_temp_base
    historical_load = 0
    
    for i, (speed, power_level) in enumerate(zip(fan_speeds, task_levels)):
        current_temp = temp_ambient + heat / heat_on_temp_base
        
        # --- 动态热容（高温时热容增大）---
        heat_on_temp = heat_on_temp_base

        temp_gap = max(0, temp_target - current_temp)
        heat_gen_boost = 1 + 2.0 * (temp_gap / temp_target)
        
        if power_level < power_threshold:
            heat_gen = heat_add_base * power_level * 0.5 * heat_gen_boost
        else:
            overload_ratio = (power_level - power_threshold) / (1 - power_threshold)
            heat_gen = heat_add_base * power_level * (1 + overload_ratio) * heat_gen_boost
        
        # --- 热生成改进 ---
        # 负载响应延迟（平滑突变负载）
        # historical_load = 0.7*historical_load + 0.3*power_level
        # effective_load = historical_load
        
        # if effective_load < power_threshold:
        #     # 低负载模式：效率降低且温度越高效率衰减越快
        #     temp_factor = np.exp(-0.03*(current_temp - 60))
        #     heat_gen = heat_add_base * effective_load * 0.4 * temp_factor
        # else:
        #     # 高负载模式：动态增益
        #     overload_ratio = (effective_load - power_threshold) / (1 - power_threshold)
        #     heat_gen = heat_add_base * effective_load * (1 + power_gain * overload_ratio)
        
        # --- 散热改进 ---
        # 风扇散热（温度越高效率略提升）
        fan_cooling = (fan_efficiency * fan_eff_ratio) * (1 - np.exp(-alpha*speed - beta*speed**2))
        fan_cooling *= (1 + 0.002*(current_temp - 65))
        
        # 协同效应：风扇加速水箱散热
        synergy = 1 + 0.5*(1 - np.exp(-0.01*speed))
        total_cooling = (fan_cooling + tank_cooling) * synergy
        
        # --- 状态更新 ---
        delta_heat = heat_gen - total_cooling
        heat += delta_heat
        T_new = temp_ambient + heat / heat_on_temp
        T.append(T_new)
    
    return np.array(T[:-1])

def robust_loss(params, fan_speeds, task_levels, T_real, T0):
    T_sim = simulate_temp(fan_speeds, task_levels, params, T0)
    errors = T_sim - T_real
    # Huber损失函数，减少异常值影响
    delta = 2.0
    loss = np.where(np.abs(errors) < delta, 
                   0.5 * errors**2,
                   delta * (np.abs(errors) - 0.5 * delta))
    return np.mean(loss)

def fit_model_enhanced(fan_speeds, task_levels, T_real):
    # 第一阶段：快速拟合主要参数
    initial_params = [
        1200,    # heat_add_base (比原值增大，反映实测高负载升温快)
        8000,    # fan_efficiency (包含水箱散热的总能力)
        1.5,     # alpha (散热曲线形状)
        0.3,     # beta 
        12000,   # heat_on_temp_base (基础热容)
        0.35,    # power_threshold (优化高低负载分界)
        80      # temp_target (高负载增益增强)
    ]
    
    phase1_bounds = [
        (100, 1000),    # heat_add_base
        (100, 10000),  # fan_efficiency
        (0.1, 5),       # alpha
        (0.01, 2),      # beta
        (1000, 20000),  # heat_on_temp
        (0.2, 0.9),     # power_threshold
        (1, 100)        # power_gain
    ]

    # 使用差分进化算法进行全局搜索
    from scipy.optimize import differential_evolution
    result = differential_evolution(
        lambda p: robust_loss(p, fan_speeds, task_levels, T_real, T_real[0]),
        bounds=phase1_bounds,
        strategy='best1bin',
        maxiter=200,
        popsize=50
    )
    
    # 第二阶段：局部精细化优化
    refined_result = minimize(
        robust_loss,
        result.x,
        args=(fan_speeds, task_levels, T_real, T_real[0]),
        method='L-BFGS-B',
        bounds=phase1_bounds
    )
    
    return refined_result.x

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
        
        best_params = fit_model_enhanced(fan_speeds, task_levels, T_real)
        print(path)
        print("Optimized Parameters:", best_params)
        
        T_sim = simulate_temp(fan_speeds, task_levels, best_params, T_real[0])
        mae = np.mean(np.abs(T_sim - T_real))
        max_error = np.max(np.abs(T_sim - T_real))

        # results = {
        #     'T_real': T_real,
        #     'T_sim': T_sim,
        #     'Absolute_Error': np.abs(T_sim - T_real),
        #     'metrics': {'MAE': mae, 'MaxError': max_error}
        # }
        # pd.DataFrame(results).to_csv("./result/enhanced_simulation_results.csv", index=False)
