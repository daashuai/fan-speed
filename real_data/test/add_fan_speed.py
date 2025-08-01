import pandas as pd
import numpy as np
from random import choices

def transfer_power_to_task_level(source_file, target_file, output_file):
    """
    将source_file中的Power(W)列值赋给target_file的task_level列
    若source数据不足，则随机抽取已有值补全
    
    参数：
        source_file: 提供Power(W)数据的CSV文件路径
        target_file: 需要添加task_level的CSV文件路径
        output_file: 结果输出文件路径
    """
    # 读取源数据（Power数据）
    source_df = pd.read_csv(source_file)
    power_values = source_df['Power(W)'].dropna().values  # 去除NaN值
    
    if len(power_values) == 0:
        raise ValueError("源文件中没有可用的Power(W)数据")
    
    # 读取目标数据
    target_df = pd.read_csv(target_file)
    target_length = len(target_df)
    
    # 判断是否需要补全数据
    if len(power_values) >= target_length:
        # 源数据足够，直接截取
        target_df['task_level'] = power_values[:target_length]
    else:
        # 源数据不足，随机抽取补全
        needed = target_length - len(power_values)
        extra_values = choices(power_values, k=needed)  # 有放回随机抽样
        target_df['task_level'] = np.concatenate([power_values, extra_values])
    
    # 保存结果
    target_df.to_csv(output_file, index=False)
    print(f"成功处理，结果保存到 {output_file}")
    print(f"分配统计：原始Power值 {len(power_values)}个，补充 {max(0, target_length - len(power_values))}个")

# 使用示例
if __name__ == "__main__":
    # 文件路径配置
    power_source = "cpu_100%.csv"    # 包含Power(W)列的文件
    target_data = "8cpu_100%_90s.csv"    # 需要添加task_level的文件
    result_file = "8cpu_100%_90s_wt.csv"  # 输出文件
    
    try:
        transfer_power_to_task_level(power_source, target_data, result_file)
        
        # 验证结果
        result_df = pd.read_csv(result_file)
        print("\n结果文件预览：")
        print(result_df[['task_level']].describe())
        
    except FileNotFoundError as e:
        print(f"文件错误：{str(e)}")
    except Exception as e:
        print(f"处理失败：{str(e)}")