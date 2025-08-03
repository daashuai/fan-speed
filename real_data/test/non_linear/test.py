import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_test_data(file_path, max_fan_rpm=4000, max_power_w=85):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    core_cols = [col for col in df.columns if col.startswith("Core")]
    df["Temp_Avg"] = df[core_cols].mean(axis=1)
    df = df[(df["Temp_Avg"] > 30) & (df["Temp_Avg"] < 120)]  
    
    fan_speeds = np.clip(df["Fan_Speed(RPM)"].values, 0, max_fan_rpm) / max_fan_rpm

    if "task_level" in df.columns:
        task_levels = np.clip(df["task_level"].values, 0, max_power_w) / max_power_w
    else:
        task_levels = np.ones(len(df)) * 0.8 
        
    T_real = df["Temp_Avg"].values
    return fan_speeds, task_levels, T_real

# def simulate_temp(fan_speeds, task_levels, params, T0, temp_ambient=50.0):
#     (heat_add_base, fan_efficiency, alpha, beta, 
#      heat_on_temp, power_threshold, power_gain) = params
    
#     # 将fan_efficiency拆分为风扇和水箱两部分（比例固定）
#     fan_eff_ratio = 0.995
#     tank_cooling = fan_efficiency * (1 - fan_eff_ratio)  # 水箱固定降温量
    
#     T = [T0]
#     heat = (T0 - temp_ambient) * heat_on_temp
    
#     for speed, power_level in zip(fan_speeds, task_levels):
#         current_temp = temp_ambient + heat / heat_on_temp
        
#         # 热生成部分保持不变
#         if power_level < power_threshold:
#             heat_gen = heat_add_base * power_level * 0.6
#         else:
#             overload_ratio = (power_level - power_threshold) / (1 - power_threshold)
#             heat_gen = heat_add_base * power_level * (1 + power_gain * overload_ratio)
        
#         # 散热部分：风扇 + 水箱
#         fan_cooling = (fan_efficiency * fan_eff_ratio) * (1 - np.exp(-alpha*speed - beta*speed**2))
#         total_cooling = fan_cooling + tank_cooling  # 总散热
        
#         delta_heat = heat_gen - total_cooling
#         heat += delta_heat
#         T_new = temp_ambient + heat / heat_on_temp
#         T.append(T_new)
    
#     return np.array(T[:-1])
def simulate_temp(fan_speeds, task_levels, params, T0, temp_ambient=50.0):
    (heat_add_base, fan_efficiency, alpha, beta, 
     heat_on_temp, power_threshold, temp_target) = params
    
    # 散热分配比例（风扇vs水箱）
    fan_eff_ratio = 0.65  # 风扇主导散热
    tank_cooling = fan_efficiency * (1 - fan_eff_ratio)
    
    T = [T0]
    heat = (T0 - temp_ambient) * heat_on_temp
    historical_load = 0
    
    for i, (speed, power_level) in enumerate(zip(fan_speeds, task_levels)):
        current_temp = temp_ambient + heat / heat_on_temp

        temp_gap = max(0, temp_target - current_temp)
        heat_gen_boost = 1 + 3.5 * (temp_gap / temp_target)
        
        if power_level < power_threshold:
            heat_gen = heat_add_base * power_level * 0.5 * heat_gen_boost
        else:
            overload_ratio = (power_level - power_threshold) / (1 - power_threshold)
            heat_gen = heat_add_base * power_level * (1 + overload_ratio) * heat_gen_boost
        
        # 风扇散热（温度越高效率略提升）
        fan_cooling = (fan_efficiency * fan_eff_ratio) * (1 - np.exp(-alpha*speed - beta*speed**2))
        fan_cooling *= (1 + 0.002*(current_temp - 55))
        
        # 协同效应：风扇加速水箱散热
        synergy = 1 + 0.5*(1 - np.exp(-0.01*speed))
        total_cooling = (fan_cooling + tank_cooling) * synergy
        
        # --- 状态更新 ---
        delta_heat = heat_gen - total_cooling
        heat += delta_heat
        T_new = temp_ambient + heat / heat_on_temp
        T.append(T_new)
    
    return np.array(T[:-1])

def loss(params, fan_speeds, task_levels, T_real, T0):
    T_sim = simulate_temp(fan_speeds, task_levels, params, T0)
    return np.mean((T_sim - T_real) ** 2)

def validate_on_testset(test_file, params):
    fan_speeds, task_levels, T_real = load_test_data(test_file)
    
    T0 = T_real[0]  
    T_sim = simulate_temp(fan_speeds, task_levels, params, T0)
    
    mse = np.mean((T_sim - T_real) ** 2)
    mae = np.mean(np.abs(T_sim - T_real))
    max_error = np.max(np.abs(T_sim - T_real))
    
    print("\n验证结果:")
    print(f"均方误差(MSE): {mse:.4f}")
    print(f"平均绝对误差(MAE): {mae:.4f}")
    print(f"最大绝对误差: {max_error:.4f}")
    
    return {
        'T_real': T_real,
        'T_sim': T_sim,
        'metrics': {'MSE': mse, 'MAE': mae, 'MaxError': max_error}
    }

if __name__ == "__main__":
    fitted_params = [2200, 1200, 0.4, 0.8, 9000, 0.8, 80]  # heat_add_base, fan_efficiency, alpha, beta, heat_on_temp, power_threshold, temp_target
    
    file = [
        "8cpu_20%_90s_wt.csv",
        "8cpu_50%_90s_wt.csv",
        "8cpu_100%_90s_wt.csv",
    ]
    
    for test_file in file:
        results = validate_on_testset(test_file, fitted_params)
        
        result_df = pd.DataFrame({
            'Real_Temp': results['T_real'],
            'Simulated_Temp': results['T_sim'],
            'Absolute_Error': np.abs(results['T_real'] - results['T_sim'])
        })
        result_df.to_csv(f'validation_results_{test_file}.csv', index=False)