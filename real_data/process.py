import pandas as pd

def add_task_level_column(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 添加新列'task_level'，值为1.0（浮点型）
    df['task_level'] = 3.4
    
    # 将修改后的数据写回原文件
    df.to_csv(file_path, index=False)
    print(f"已成功在文件 {file_path} 中添加 task_level 列")

if __name__ == "__main__":
    add_task_level_column("8cpu_100%_90s.csv")