import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import os
import pickle
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# 0. 定义CPU数据集类（包含温度信息）
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
# 1. 配置参数（集中管理）
# =====================
class Config:
    """模型训练配置参数"""
    BATCH_SIZE = 32  # 批处理大小
    EPOCHS = 200  # 训练轮数
    LEARNING_RATE = 0.001  # 学习率
    HIDDEN_DIM = 64  # 隐藏层维度
    INPUT_DIM = 2  # 输入特征维度（功耗+风扇转速）
    OUTPUT_DIM = 1  # 输出维度（温度变化率ΔT）
    TRANSFORMER_NHEAD = 2  # Transformer多头注意力头数
    TRANSFORMER_LAYERS = 2  # Transformer编码器层数
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备
    DATA_DIR = "../datasets"  # 数据集保存目录
    SEQ_LEN = 10  # 时间窗口长度，需与数据集创建时一致


# =====================
# 2. 模型定义
# =====================
class LSTMModel(nn.Module):
    """LSTM模型用于时间序列预测"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        return self.fc(out[:, -1, :])  # 取最后一个时间步的输出


class GRUModel(nn.Module):
    """GRU模型用于时间序列预测"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)  # (batch_size, seq_len, hidden_dim)
        return self.fc(out[:, -1, :])  # 取最后一个时间步的输出


class TransformerModel(nn.Module):
    """Transformer模型用于时间序列预测"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, nhead: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)  # 映射到Transformer的d_model维度
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)  # (batch_size, seq_len, hidden_dim)
        out = self.transformer(x)  # (batch_size, seq_len, hidden_dim)
        return self.fc(out[:, -1, :])  # 取最后一个时间步的输出


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: Config
) -> Tuple[nn.Module, List[float], List[float]]:
    """训练模型并返回最优模型（测试损失最低）及训练/测试损失"""
    model.to(config.DEVICE)
    criterion = nn.MSELoss()  # 回归任务用MSE
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    train_losses: List[float] = []
    test_losses: List[float] = []
    
    # 初始化最佳模型参数和最佳损失
    best_test_loss = float('inf')
    best_model_weights = None  # 用于保存最佳权重
    
    for epoch in range(config.EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for x_batch, y_batch, true_temps_batch in train_loader:
            x_batch, y_batch = x_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_batch.size(0)
        
        train_loss_avg = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss_avg)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, true_temps_batch in test_loader:
                x_batch, y_batch = x_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * x_batch.size(0)
        
        test_loss_avg = test_loss / len(test_loader.dataset)
        test_losses.append(test_loss_avg)
        
        # 保存最佳模型权重（当当前测试损失低于历史最佳时）
        if test_loss_avg < best_test_loss:
            best_test_loss = test_loss_avg
            # 深拷贝当前模型权重（避免后续训练覆盖）
            best_model_weights = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"Epoch {epoch+1:02d} - 测试损失更新为最佳: {best_test_loss:.6f}")
        
        # 打印日志
        print(f"Epoch {epoch+1:02d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss_avg:.6f} | "
              f"Test Loss: {test_loss_avg:.6f}")
    
    # 加载最佳权重到模型
    model.load_state_dict(best_model_weights)
    return model, train_losses, test_losses, best_test_loss


# =====================
# 4. 温度还原与可视化
# =====================
def infer_temperature(
    model: nn.Module,
    test_dataset: torch.utils.data.Dataset,
    scaler_y: StandardScaler,
    config: Config
) -> Tuple[List[float], List[float]]:
    """从预测的ΔT还原温度曲线，使用数据集中的真实温度对比"""
    model.eval()
    model.to(config.DEVICE)
    
    pred_temps = []  # 预测的目标温度
    true_temps = []  # 真实的目标温度
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            # 从数据集获取三个返回值：特征、目标、温度序列
            x, _, true_temp_seq = test_dataset[i]
            # true_temp_seq形状: (seq_len+1,)，包含[窗口温度, 目标温度]
            
            # 预测ΔT（标准化后）并反标准化
            x = x.unsqueeze(0).to(config.DEVICE)  # 增加batch维度 (1, seq_len, 2)
            delta_t_pred_scaled = model(x).cpu().numpy()[0][0]  # 预测的ΔT（标准化）
            delta_t_pred = scaler_y.inverse_transform([[delta_t_pred_scaled]])[0][0]  # 反标准化
            
            # 计算预测温度：窗口最后一个温度 + 预测的ΔT
            # 窗口最后一个温度是true_temp_seq[-2]（因为最后一个是目标温度）
            pred_temp = true_temp_seq[-2] + delta_t_pred
            pred_temps.append(pred_temp)
            
            # 真实目标温度是温度序列的最后一个值
            true_temps.append(true_temp_seq[-1])
    
    return pred_temps, true_temps


# =====================
# 5. 主函数（串联所有流程）
# =====================
def main():
    config = Config()
    print(f"使用设备: {config.DEVICE}")
    
    # 1. 加载数据集（包含特征、目标和温度序列）
    print("\n加载数据集...")
    try:
        with open(f"{config.DATA_DIR}/train.pkl", "rb") as f:
            train_dataset = pickle.load(f)
        with open(f"{config.DATA_DIR}/test.pkl", "rb") as f:
            test_dataset = pickle.load(f)
    except FileNotFoundError:
        raise ValueError(f"未找到数据集文件，请检查 {config.DATA_DIR} 目录是否存在")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True  # 加速GPU传输
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )
    print(f"训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")
    
    # 2. 加载标准化器
    scaler_y = joblib.load(f"{config.DATA_DIR}/scaler_Y.pkl")
    
    # 3. 定义模型并训练
    print("\n初始化模型...")
    models: Dict[str, nn.Module] = {
        "LSTM": LSTMModel(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM),
        "GRU": GRUModel(config.INPUT_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM),
        "Transformer": TransformerModel(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            nhead=config.TRANSFORMER_NHEAD,
            num_layers=config.TRANSFORMER_LAYERS
        )
    }
    
    results: Dict[str, Tuple[nn.Module, List[float], List[float]]] = {}
    for name, model in models.items():
        print(f"\n===== 训练 {name} 模型 =====")
        trained_model, train_loss, test_loss, best_test_loss = train_model(
            model, train_loader, test_loader, config
        )
        results[name] = (trained_model, train_loss, test_loss, best_test_loss)
    

    # 4. Plot loss comparison curve
    plt.figure(figsize=(10, 6))
    for name, (_, _, test_loss, best_test_loss) in results.items():
        plt.plot(test_loss, label=f"{name} Test Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("Test Loss Comparison of Different Models", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig("model_loss_comparison.png")
    plt.show()
    
    # 5. Temperature prediction and visualization with the best model

    best_model_name = min(results, key=lambda k: results[k][3])
    best_model = results[best_model_name][0]
    print(f"\nUsing the best model {best_model_name} for temperature prediction...")
    
    pred_temps, real_temps = infer_temperature(
        best_model, test_dataset, scaler_y, config
    )
    
    # Plot temperature comparison curve with error statistics
    plt.figure(figsize=(14, 8))
    
    # Plot temperature curves
    plt.plot(real_temps, label="True Temperature", linewidth=2)
    plt.plot(pred_temps, label=f"{best_model_name} Predicted Temperature", linestyle="--", linewidth=2)
    
    # Calculate errors
    errors = np.array(pred_temps) - np.array(real_temps)
    max_error = np.max(np.abs(errors))  # Maximum absolute error
    min_error = np.min(np.abs(errors))  # Minimum absolute error
    avg_error = np.mean(np.abs(errors))  # Average absolute error
    
    # Add error statistics text box
    error_text = (f"Error Statistics:\n"
                  f"Max Absolute Error: {max_error:.2f} °C\n"
                  f"Min Absolute Error: {min_error:.2f} °C\n"
                  f"Avg Absolute Error: {avg_error:.2f} °C")
    
    # Place text box (position can be adjusted as needed)
    plt.text(0.02, 0.98, error_text, 
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Chart labels
    plt.xlabel("Sample Index (Time Sequence)", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.title(f"{best_model_name} Temperature Prediction vs True Values", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # Save and display
    plt.savefig("temperature_prediction_with_errors.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print error information to console
    print(f"\nError Statistics:")
    print(f"Maximum Absolute Error: {max_error:.2f} °C")
    print(f"Minimum Absolute Error: {min_error:.2f} °C")
    print(f"Average Absolute Error: {avg_error:.2f} °C")


if __name__ == "__main__":
    main()

