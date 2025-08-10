import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from glob import glob
import joblib
import os
import pickle  # 用于保存Dataset对象

# =====================
# 1. 定义CPU数据集类（包含温度信息）
# =====================
class CPUDataset(Dataset):
    """CPU温度变化率预测数据集（包含真实温度信息）"""
    def __init__(self, features, targets, true_temps):
        """
        Args:
            features: 输入特征张量，形状为 (样本数, SEQ_LEN, 2)
            targets: 目标变量张量，形状为 (样本数, 1)
            true_temps: 真实温度序列，形状为 (样本数, SEQ_LEN+1)
                        （前SEQ_LEN个为输入窗口的真实温度，最后1个是Label的温度）
        """
        self.features = features  # 标准化后的特征
        self.targets = targets    # 标准化后的温度变化率
        self.true_temps = true_temps  # 真实温度序列

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (self.features[idx], self.targets[idx],
                self.true_temps[idx]) 


# =====================
# 2. 处理单个文件生成样本（包含温度信息）
# =====================
def process_file(file_path, seq_len=10):
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 提取特征
    features = df[["Power(W)", "Fan(RPM)"]].values
    
    # 计算平均温度和温度变化率
    temp_cols = [col for col in df.columns if "Temp_core" in col]
    avg_temp = df[temp_cols].mean(axis=1).values  # 原始平均温度序列
    delta_t = avg_temp[1:] - avg_temp[:-1]
    delta_t = np.insert(delta_t, 0, 0.0)  # 第一个时间步变化率为0
    
    # =====================
    # 1. 先对整个数据集做滑动窗口分割（生成所有可能的样本）
    # =====================
    X_all, Y_all = [], []
    true_temps_all = []  # 保存每个样本对应的真实温度序列
    
    # 遍历所有可能的窗口起始点
    for i in range(len(features) - seq_len):
        X_all.append(features[i:i+seq_len])  # 窗口内的特征 (seq_len, 2)
        Y_all.append(delta_t[i+seq_len])     # 窗口结束后的ΔT (1,)
        
        # 真实温度序列：输入窗口温度（i到i+seq_len-1）+ 下一步的目标温度（i+seq_len）
        true_temp_seq = avg_temp[i : i+seq_len+1]  # 长度为seq_len+1
        true_temps_all.append(true_temp_seq)
    
    # 转换为numpy数组
    X_all = np.array(X_all)  # 形状: (总样本数, seq_len, 2)
    Y_all = np.array(Y_all).reshape(-1, 1)  # 形状: (总样本数, 1)
    true_temps_all = np.array(true_temps_all)  # 形状: (总样本数, seq_len+1)
    
    # =====================
    # 2. 再对样本集进行训练/测试划分（按时间顺序）
    # =====================
    # 计算划分索引（总样本的70%作为训练集）
    split_idx = int(len(X_all) * 0.7)
    
    # 划分训练集和测试集（包含温度信息）
    X_train, Y_train = X_all[:split_idx], Y_all[:split_idx]
    X_test, Y_test = X_all[split_idx:], Y_all[split_idx:]
    
    true_temps_train = true_temps_all[:split_idx]
    true_temps_test = true_temps_all[split_idx:]
    
    
    return (X_train, Y_train, true_temps_train, 
            X_test, Y_test, true_temps_test )


# =====================
# 3. 主函数：生成并保存包含温度信息的Dataset
# =====================
def main():
    SEQ_LEN = 10
    OUTPUT_DIR = "../datasets"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有文件
    file_paths = glob("../../real_data/train/cpu_*.csv")
    if not file_paths:
        raise ValueError("未找到CPU数据文件，请检查路径")
    
    # 收集所有文件的原始样本（包含温度信息）
    all_X_train, all_Y_train = [], []
    all_true_temps_train = []
    all_X_test, all_Y_test = [], []
    all_true_temps_test = []
    
    for i, file_path in enumerate(file_paths):
        print(f"处理文件 {i+1}/{len(file_paths)}: {file_path}")
        (X_train, Y_train, true_temps_train,
         X_test, Y_test, true_temps_test) = process_file(file_path, SEQ_LEN)
        
        all_X_train.append(X_train)
        all_Y_train.append(Y_train)
        all_true_temps_train.append(true_temps_train)
        
        all_X_test.append(X_test)
        all_Y_test.append(Y_test)
        all_true_temps_test.append(true_temps_test)
    
    # 合并所有样本（转换为numpy数组）
    X_train = np.concatenate(all_X_train, axis=0)  # (总样本数, SEQ_LEN, 2)
    Y_train = np.concatenate(all_Y_train, axis=0)  # (总样本数, 1)
    true_temps_train = np.concatenate(all_true_temps_train, axis=0)  # (总样本数, SEQ_LEN+1)
    
    X_test = np.concatenate(all_X_test, axis=0)
    Y_test = np.concatenate(all_Y_test, axis=0)
    true_temps_test = np.concatenate(all_true_temps_test, axis=0)
    
    print(f"合并后样本量 - 训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")
    
    # 全局标准化
    # 特征标准化（按特征维度处理）
    scaler_X = StandardScaler()
    n_samples, seq_len, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)  # 展平为二维用于标准化
    scaler_X.fit(X_train_reshaped)
    
    # 目标值标准化
    scaler_Y = StandardScaler()
    scaler_Y.fit(Y_train)
    
    # 应用标准化
    X_train_scaled = scaler_X.transform(X_train.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape[0], seq_len, n_features)
    
    Y_train_scaled = scaler_Y.transform(Y_train)
    Y_test_scaled = scaler_Y.transform(Y_test)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32)
    true_temps_train_tensor = torch.tensor(true_temps_train, dtype=torch.float32)
    
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test_scaled, dtype=torch.float32)
    true_temps_test_tensor = torch.tensor(true_temps_test, dtype=torch.float32)
    
    # 创建包含温度信息的Dataset对象
    train_dataset = CPUDataset(
        X_train_tensor, Y_train_tensor, true_temps_train_tensor 
    )
    test_dataset = CPUDataset(
        X_test_tensor, Y_test_tensor, true_temps_test_tensor 
    )
    
    # 保存Dataset对象（使用pickle）
    with open(f"{OUTPUT_DIR}/train.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    
    with open(f"{OUTPUT_DIR}/test.pkl", "wb") as f:
        pickle.dump(test_dataset, f)
    
    # 保存标准化器
    joblib.dump(scaler_X, f"{OUTPUT_DIR}/scaler_X.pkl")
    joblib.dump(scaler_Y, f"{OUTPUT_DIR}/scaler_Y.pkl")
    
    print(f"数据集保存完成！位置：{OUTPUT_DIR}")
    print(f"训练集样本数：{len(train_dataset)}, 测试集样本数：{len(test_dataset)}")


if __name__ == "__main__":
    main()

