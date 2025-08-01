import pandas as pd

def normalize_task_level(input_file, output_file, original_min=3.08, original_max=84.74):
    """
    对CSV文件的task_level列进行归一化
    
    参数：
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        original_min: 原始最小值（默认3.08）
        original_max: 原始最大值（默认84.74）
    """
    # 读取数据
    df = pd.read_csv(input_file)
    
    # 检查task_level列是否存在
    if 'task_level' not in df.columns:
        raise ValueError("输入文件中没有task_level列")
    
    # 归一化公式：(x - min) / (max - min)
    df['normalized_task'] = (df['task_level'] - original_min) / (original_max - original_min)
    
    # 确保结果在[0,1]范围内（处理可能的浮点误差）
    df['normalized_task'] = df['normalized_task'].clip(0, 1)
    
    # 保存结果
    df.to_csv(output_file, index=False)
    print(f"归一化完成，结果已保存到 {output_file}")
    
    # 打印统计信息
    print("\n归一化结果统计：")
    print(f"原始最小值: {original_min}")
    print(f"原始最大值: {original_max}")
    print(f"归一化后最小值: {df['normalized_task'].min():.6f}")
    print(f"归一化后最大值: {df['normalized_task'].max():.6f}")
    print(f"均值: {df['normalized_task'].mean():.4f}")

# 使用示例
if __name__ == "__main__":
    input_csv = "8cpu_100%_300s_wt.csv"  # 替换为你的输入文件
    output_csv = "8cpu_100%_300s_wt_normal.csv"  # 输出文件路径
    
    try:
        normalize_task_level(input_csv, output_csv)
        
        # 验证结果
        print("\n结果文件前5行预览：")
        result = pd.read_csv(output_csv)
        print(result.head())
        
    except Exception as e:
        print(f"处理出错: {str(e)}")