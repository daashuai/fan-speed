import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载测试集数据
def load_test_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    # 计算核心平均温度
    core_cols = [col for col in df.columns if col.startswith("Core")]
    df["Temp_Avg"] = df[core_cols].mean(axis=1)
    
    # 提取所需数据
    timestamps = pd.to_datetime(df["Timestamp"])
    fan_speeds = df["Fan_Speed(RPM)"].values
    task_levels = df["task_level"].values
    T_real = df["Temp_Avg"].values
    
    return timestamps, fan_speeds, task_levels, T_real

# 2. 温度模拟函数（保持不变）
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

# 3. 验证函数
def validate_on_testset(test_file, params):
    # 加载测试数据
    timestamps, fan_speeds, task_levels, T_real = load_test_data(test_file)
    
    # 使用拟合参数进行模拟
    T0 = T_real[0]  # 使用测试集的初始温度
    T_sim = simulate_temp(fan_speeds, task_levels, params, T0)
    
    # 计算评估指标
    mse = np.mean((T_sim - T_real) ** 2)
    mae = np.mean(np.abs(T_sim - T_real))
    max_error = np.max(np.abs(T_sim - T_real))
    
    # 打印结果
    print("\n验证结果:")
    print(f"均方误差(MSE): {mse:.4f}")
    print(f"平均绝对误差(MAE): {mae:.4f}")
    print(f"最大绝对误差: {max_error:.4f}")
    
    # 绘制对比图
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, T_real, 'b-', label="real_temp")
    plt.plot(timestamps, T_sim, 'r--', label="simulate_temp")
    plt.fill_between(timestamps, T_sim-2, T_sim+2, color='pink', alpha=0.3, label="±2°C误差带")
    plt.title("温度模拟验证 (测试集)")
    plt.xlabel("时间")
    plt.ylabel("温度 (°C)")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("temperature_validation.png", dpi=300)
    plt.show()
    
    return {
        'timestamps': timestamps,
        'T_real': T_real,
        'T_sim': T_sim,
        'metrics': {'MSE': mse, 'MAE': mae, 'MaxError': max_error}
    }

# 4. 主程序
if __name__ == "__main__":
    # 拟合好的参数（来自训练集）
    fitted_params = [35.30, 645.85, 881.69, 1060.54]  # heat_add, heat_remove, fan_effect_modifer, heat_on_temp
    
    # 测试集文件路径
    test_file = "8cpu_20%_90s_wt.csv"  # 替换为您的测试集路径
    
    # 运行验证
    results = validate_on_testset(test_file, fitted_params)
    
    # 可选：保存详细结果到CSV
    result_df = pd.DataFrame({
        'Timestamp': results['timestamps'],
        'Real_Temp': results['T_real'],
        'Simulated_Temp': results['T_sim'],
        'Absolute_Error': np.abs(results['T_real'] - results['T_sim'])
    })
    result_df.to_csv("validation_results_20%_90s.csv", index=False)