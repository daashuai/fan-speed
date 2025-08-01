import pandas as pd
import numpy as np
from scipy.optimize import minimize

KNOWN_PARAMS = {
    'heat_add': 422,
    'heat_remove': 400,
    'fan_effect_modifer': 963,
    'heat_on_temp': 1005
}

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    core_cols = [col for col in df.columns if col.startswith("Core")]
    df["Temp_Avg"] = df[core_cols].mean(axis=1)
    fan_speeds = df["Fan_Speed(RPM)"].values
    T_real = df["Temp_Avg"].values
    return fan_speeds, T_real

def simulate_temp(fan_speeds, task_levels, T0, temp_ambient=20.0):
    heat_add = KNOWN_PARAMS['heat_add']
    heat_remove = KNOWN_PARAMS['heat_remove']
    fan_effect_modifer = KNOWN_PARAMS['fan_effect_modifer']
    heat_on_temp = KNOWN_PARAMS['heat_on_temp']
    
    T = [T0]
    heat = (T0 - temp_ambient) * heat_on_temp
    for speed, task_level in zip(fan_speeds, task_levels):
        delta_heat = heat_add * task_level - heat_remove * (speed / fan_effect_modifer)
        heat += delta_heat
        T_new = temp_ambient + heat / heat_on_temp
        T.append(T_new)
    return np.array(T[:-1])

def loss(task_levels, fan_speeds, T_real, T0):
    T_sim = simulate_temp(fan_speeds, task_levels, T0)
    return np.mean((T_sim - T_real) ** 2)

def optimize_task_levels(fan_speeds, T_real):
    initial_guess = np.ones(len(fan_speeds)) * 1.0
    
    bounds = [(0.1, 10.0)] * len(fan_speeds)
    
    result = minimize(loss, initial_guess,
                     args=(fan_speeds, T_real, T_real[0]),
                     method='L-BFGS-B',
                     bounds=bounds,
                     options={'maxiter': 1000})
    
    return result.x

def main():
    input_file = "8cpu_20%_300s.csv"  # 替换为你的文件路径
    output_file = "task_level/optimized_data_8cpu_20%_300s.csv"
    
    # 加载数据
    fan_speeds, T_real = load_and_process_data(input_file)
    print(f"数据记录数: {len(T_real)}")
    
    # 优化task_levels
    print("开始优化task_levels...")
    optimized_task_levels = optimize_task_levels(fan_speeds, T_real)
    
    # 计算最终误差
    final_mse = loss(optimized_task_levels, fan_speeds, T_real, T_real[0])
    print(f"优化完成，最终MSE: {final_mse:.4f}")
    
    # 保存结果
    df = pd.read_csv(input_file)
    df['optimized_task_level'] = optimized_task_levels
    df.to_csv(output_file, index=False)
    print(f"结果已保存到 {output_file}")
    
    # 打印统计信息
    print("\n优化后的task_levels统计:")
    print(f"  平均值: {np.mean(optimized_task_levels):.2f}")
    print(f"  标准差: {np.std(optimized_task_levels):.2f}")
    print(f"  最小值: {np.min(optimized_task_levels):.2f}")
    print(f"  最大值: {np.max(optimized_task_levels):.2f}")

if __name__ == "__main__":
    main()