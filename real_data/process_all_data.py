import pandas as pd

def add_task_level(input_file, output_file=None):
    """
    为CSV文件添加task_level列并按行范围赋值
    :param input_file: 输入CSV文件路径
    :param output_file: 输出文件路径（默认覆盖原文件）
    """
    df = pd.read_csv(input_file)

    df['task_level'] = 0.0  

    df.loc[0:298, 'task_level'] = 1.4   
    df.loc[299:600, 'task_level'] = 2.4  
    df.loc[601:, 'task_level'] = 3.4      
    
    output_file = output_file or input_file 
    df.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存到 {output_file}")
    print(f"任务级别分布:\n{df['task_level'].value_counts()}")

if __name__ == "__main__":
    input_csv = "all_data.csv" 
    add_task_level(input_csv)