import pandas as pd
import numpy as np
from scipy.optimize import minimize, basinhopping, dual_annealing
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
    fan_speeds = df["Fan_Speed(RPM)"].values
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
    # result = minimize(loss, initial_guess,
    #              args=(fan_speeds, task_levels, T_real, T_real[0]),
    #              method='L-BFGS-B',
    #              bounds=[(0,None), (0,None), (0,None), (0,None)],  
    #              options={'ftol': 1e-8, 'maxiter': 1000}) 
    
    result = minimize(loss, initial_guess, args=(fan_speeds, task_levels, T_real, T_real[0]), method='Nelder-Mead')
    # result = dual_annealing(loss, 
    #             bounds=[(1,2000), (1,2000), (1,2000), (1,2000)],  # 扩大搜索范围
    #             args=(fan_speeds, task_levels, T_real, T_real[0]),
    #             maxiter=1000)
    return result.x

def save_results(file_path, params, mse):
    """将结果保存到文件"""
    with open("fitted_parameters.txt", "a") as f:
        f.write(f"\n文件: {file_path}\n")
        f.write(f"  heat_add: {params[0]:.2f}\n")
        f.write(f"  heat_remove: {params[1]:.2f}\n")
        f.write(f"  fan_effect_modifer: {params[2]:.2f}\n")
        f.write(f"  heat_on_temp: {params[3]:.2f}\n")
        f.write(f"  MSE: {mse:.4f}\n")
        f.write("-" * 40 + "\n")

def main():
    data_files = [
        "8cpu_20%_300s_wt.csv",
        "8cpu_50%_300s_wt.csv",
        "8cpu_100%_300s_wt.csv",
        "8cpu_20%_300s_wt_normal.csv",
        "8cpu_50%_300s_wt_normal.csv",
        "8cpu_100%_300s_wt_normal.csv",
    ]
    
    with open("fitted_parameters.txt", "w") as f:
        f.write("物理参数拟合结果\n")
        f.write("=" * 40 + "\n")
    
    for file_path in data_files:
        print(f"\n处理文件: {file_path}")
        try:
            fan_speeds, task_levels, T_real = load_and_process_data(file_path)
            print(f"数据记录数: {len(T_real)}")
            
            # 拟合模型参数（仅优化4个物理参数）
            params = fit_model(fan_speeds, task_levels, T_real)
            
            # 计算拟合误差
            T_sim = simulate_temp(fan_speeds, task_levels, params, T_real[0])
            mse = np.mean((T_sim - T_real) ** 2)
            
            # 打印并保存结果
            print("\n拟合的物理参数:")
            print(f"  heat_add: {params[0]:.2f}")
            print(f"  heat_remove: {params[1]:.2f}")
            print(f"  fan_effect_modifer: {params[2]:.2f}")
            print(f"  heat_on_temp: {params[3]:.2f}")
            print(f"\n均方误差(MSE): {mse:.4f}")
            
            save_results(file_path, params, mse)
            
        except Exception as e:
            print(f"处理出错: {e}")
            with open("fitted_parameters.txt", "a") as f:
                f.write(f"处理 {file_path} 时出错: {e}\n")
    
    print("\n所有处理完成，结果已保存到 fitted_parameters.txt")

if __name__ == "__main__":
    main()