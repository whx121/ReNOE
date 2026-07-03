import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import List, Tuple, Optional

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


class DNNModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layers_config: List[int] = [128, 64, 32],
                 learning_rate: float = 0.001, dropout_rate: float = 0.2, l2_regularization: float = 0.01):
        """
        初始化 DNN 模型
        :param input_dim: 输入特征的维度
        :param output_dim: 输出维度
        :param layers_config: 每层的神经元数量
        :param learning_rate: 学习率
        :param dropout_rate: Dropout 的比例
        :param l2_regularization: L2 正则化系数
        """
        super(DNNModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_config = layers_config
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_regularization = l2_regularization

        # 检查CUDA是否可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 构建网络层
        self.layers = self._build_layers()

        # 初始化优化器和损失函数
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=l2_regularization)
        self.criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.2, patience=40, min_lr=1e-10
        )

        # 移动模型到设备
        self.to(self.device)

    def _build_layers(self) -> nn.ModuleList:
        """构建神经网络层"""
        layers = nn.ModuleList()

        # 输入层到第一个隐藏层
        prev_dim = self.input_dim

        for units in self.layers_config:
            # 全连接层
            layers.append(nn.Linear(prev_dim, units))
            # LeakyReLU激活函数
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            # 批归一化
            layers.append(nn.BatchNorm1d(units))
            # Dropout
            layers.append(nn.Dropout(self.dropout_rate))

            prev_dim = units

        # 输出层
        layers.append(nn.Linear(prev_dim, self.output_dim))

        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for layer in self.layers:
            x = layer(x)
        return x

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    epochs: int = 100, batch_size: int = 32) -> dict:
        """训练模型"""
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            print("tqdm not installed. Install it with: pip install tqdm")
            print("Using basic progress display...")
            use_tqdm = False

        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 训练历史记录
        history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 122  # 对应TensorFlow版本的patience

        print(f"Starting training for {epochs} epochs...")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print(f"Batch size: {batch_size}, Batches per epoch: {len(train_loader)}")
        print("-" * 80)

        # 创建epoch进度条
        if use_tqdm:
            epoch_pbar = tqdm(range(epochs), desc="Training Progress", position=0)
        else:
            epoch_pbar = range(epochs)

        for epoch in epoch_pbar:
            # 训练模式
            self.train()
            train_losses = []
            train_maes = []

            # 创建batch进度条
            if use_tqdm:
                batch_pbar = tqdm(train_loader,
                                  desc=f"Epoch {epoch + 1}/{epochs}",
                                  leave=False,
                                  position=1,
                                  ncols=100)
            else:
                batch_pbar = train_loader
                print(f"Epoch {epoch + 1}/{epochs}: ", end="")

            batch_count = 0
            for batch_X, batch_y in batch_pbar:
                self.optimizer.zero_grad()

                # 前向传播
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                mae = self.mae_criterion(outputs, batch_y)

                # 反向传播
                loss.backward()
                self.optimizer.step()

                current_loss = loss.item()
                current_mae = mae.item()
                train_losses.append(current_loss)
                train_maes.append(current_mae)

                # 更新进度条显示
                if use_tqdm:
                    batch_pbar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'MAE': f'{current_mae:.4f}',
                        'Avg_Loss': f'{np.mean(train_losses):.4f}'
                    })
                else:
                    # 简单进度显示
                    batch_count += 1
                    if batch_count % max(1, len(train_loader) // 10) == 0:
                        print(f"[{batch_count}/{len(train_loader)}]", end="")

            if not use_tqdm:
                print()  # 换行

            # 验证模式
            self.eval()
            with torch.no_grad():
                val_outputs = self(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
                val_mae = self.mae_criterion(val_outputs, y_val_tensor).item()

            # 记录历史
            avg_train_loss = np.mean(train_losses)
            avg_train_mae = np.mean(train_maes)

            history['loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['mae'].append(avg_train_mae)
            history['val_mae'].append(val_mae)

            # 学习率调度
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 检查是否有学习率变化
            lr_changed = abs(old_lr - current_lr) > 1e-10

            # 早停检查
            is_best = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                is_best = True
                # 保存最佳权重
                self.best_state_dict = self.state_dict().copy()
            else:
                patience_counter += 1

            # 更新epoch进度条
            if use_tqdm:
                status_info = {
                    'T_Loss': f'{avg_train_loss:.4f}',
                    'V_Loss': f'{val_loss:.4f}',
                    'T_MAE': f'{avg_train_mae:.4f}',
                    'V_MAE': f'{val_mae:.4f}',
                    'LR': f'{current_lr:.2e}',
                    'Best': '✓' if is_best else f'{patience_counter}',
                }
                if lr_changed:
                    status_info['LR_Change'] = '↓'

                epoch_pbar.set_postfix(status_info)
            else:
                # 简单显示
                status = f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                status += f"Train MAE: {avg_train_mae:.4f} | Val MAE: {val_mae:.4f} | "
                status += f"LR: {current_lr:.2e}"
                if is_best:
                    status += " | ★ NEW BEST"
                elif lr_changed:
                    status += " | LR REDUCED"
                else:
                    status += f" | Patience: {patience_counter}/{patience}"
                print(status)

            # 早停检查
            if patience_counter >= patience:
                if use_tqdm:
                    epoch_pbar.close()
                print(f"\n🛑 Early stopping at epoch {epoch + 1}")
                print(f"Best validation loss: {best_val_loss:.6f}")
                # 恢复最佳权重
                self.load_state_dict(self.best_state_dict)
                break

        if use_tqdm:
            epoch_pbar.close()

        print("\n" + "=" * 80)
        print("🎉 Training completed!")
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Total epochs: {len(history['loss'])}")
        print("=" * 80)

        return history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """评估模型"""
        self.eval()

        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)

        with torch.no_grad():
            predictions = self(X_test_tensor)
            test_loss = self.criterion(predictions, y_test_tensor).item()
            test_mae = self.mae_criterion(predictions, y_test_tensor).item()

        print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
        return test_loss, test_mae

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测结果"""
        self.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self(X_tensor)

        return predictions.cpu().numpy()

    def compute_jacobian(self, X: np.ndarray) -> np.ndarray:
        """
        计算神经网络输出对输入的雅各比矩阵
        :param X: 输入数据，形状为 (batch_size, input_dim)
        :return: 雅各比矩阵，形状为 (batch_size, output_dim, input_dim)
        """
        self.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)
        X_tensor.requires_grad_(True)

        # 前向传播
        outputs = self(X_tensor)

        # 计算雅各比矩阵
        batch_size, output_dim = outputs.shape
        input_dim = X_tensor.shape[1]

        jacobian = torch.zeros(batch_size, output_dim, input_dim).to(self.device)

        for i in range(output_dim):
            for j in range(batch_size):
                # 创建梯度向量（只有当前输出元素为1，其他为0）
                grad_outputs = torch.zeros_like(outputs)
                grad_outputs[j, i] = 1.0

                # 计算梯度
                grads = torch.autograd.grad(
                    outputs=outputs,
                    inputs=X_tensor,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=False
                )[0]

                jacobian[j, i, :] = grads[j, :]

        return jacobian.detach().cpu().numpy()

    def save_model(self, path: str):
        """保存模型"""
        # 保存整个模型状态
        model_state = {
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'layers_config': self.layers_config,
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate,
            'l2_regularization': self.l2_regularization
        }
        torch.save(model_state, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load_model(path: str) -> 'DNNModel':
        """加载已保存的模型"""
        # 加载模型状态
        model_state = torch.load(path, map_location='cpu')

        # 重新创建模型
        model = DNNModel(
            input_dim=model_state['input_dim'],
            output_dim=model_state['output_dim'],
            layers_config=model_state['layers_config'],
            learning_rate=model_state['learning_rate'],
            dropout_rate=model_state['dropout_rate'],
            l2_regularization=model_state['l2_regularization']
        )

        # 加载权重
        model.load_state_dict(model_state['state_dict'])

        print(f"Model loaded from {path}")
        return model

import os
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # 用于保存和加载 scaler

# 假设 DNNModel 已经正确定义和导入
# from pytorch_dnn_model import DNNModel

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查tqdm是否安装（用于显示训练进度条）
try:
    from tqdm import tqdm
    print("✓ tqdm found - will show detailed progress bars")
except ImportError:
    print("⚠️  tqdm not found - install with: pip install tqdm")
    print("   (Training will still work but with basic progress display)")

# ========================
# 参数配置
# ========================
feature_columns = ["vc_BB", "vc_Urban", "vc_Ocean", "vc_Dust"]
output_columns = ["AOD_560nm", "SSA_560nm", "AAOD_560nm", "mr", "mi", "aod_fine", "aod_coarse"]
# model_para_dir = '/media/amers/2E42853942850735/whx/Doctor/NNOE_polder/forward/aersol_param/'

print(f"Feature columns: {feature_columns}")
print(f"Output columns: {output_columns}")
print(f"Number of inputs: {len(feature_columns)}")
print(f"Number of outputs: {len(output_columns)}")
print()

# ========================
# 1. 读取数据
# ========================
data_path = '/media/amers/WHX/polder_simulation_results/train_data_for_polder.parquet'
data = pq.read_table(data_path).to_pandas()

# 获取所有特征和目标数据
X_all = data[feature_columns].values
y_all = data[output_columns].values

print(f"Total data shape: X={X_all.shape}, y={y_all.shape}")

# ========================
# 2. 数据划分（与验证代码保持一致）
# ========================
# 按训练时的划分方式：先划分出测试集，再从剩余数据中划分出验证集
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.4, random_state=40
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")
print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
print()

# ========================
# 3. 特征归一化
# ========================
scaler_features = StandardScaler()
X_train_scaled = scaler_features.fit_transform(X_train)
X_val_scaled = scaler_features.transform(X_val)
X_test_scaled = scaler_features.transform(X_test)

# 保存特征的 scaler
joblib.dump(scaler_features, "scaler_features_aerosol.pkl")
print("Features StandardScaler saved to scaler_features_aerosol.pkl")

# ========================
# 4. 目标变量归一化
# ========================
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

# 保存目标变量的 scaler
joblib.dump(scaler_y, "scaler_y_aerosol.pkl")
print("Target StandardScaler saved to scaler_y_aerosol.pkl")
print()

# ========================
# 5. 初始化并训练模型
# ========================
print("Initializing PyTorch DNN model for multi-output...")
model = DNNModel(
    input_dim=X_train_scaled.shape[1],  # 4个输入特征
    output_dim=y_train_scaled.shape[1],  # 7个输出
    layers_config=[1024, 256, 128],  # 可以根据需要调整
    learning_rate=0.001,
    dropout_rate=0.002,
    l2_regularization=0.001
)

print(f"Model architecture:")
print(f"Input dimension: {X_train_scaled.shape[1]}")
print(f"Output dimension: {y_train_scaled.shape[1]}")
print(f"Hidden layers: [512, 256, 128, 64]")
print(f"Learning rate: 0.001")
print(f"Dropout rate: 0.2")
print(f"L2 regularization: 0.01")
print()

# 训练模型
print("Starting model training...")
history = model.train_model(
    X_train_scaled, y_train_scaled,
    X_val_scaled, y_val_scaled,
    epochs=1000,
    batch_size=2048
)

print("Training completed!")
print()

# ========================
# 6. 模型评估
# ========================
print("\n📊 Evaluating model performance...")
print("-" * 50)
test_loss, test_mae = model.evaluate(X_test_scaled, y_test_scaled)

# 在原始尺度上评估
print("\n📈 Evaluating on original scale...")
y_pred_scaled = model.predict(X_test_scaled)
y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = y_test  # 原始测试数据

# 计算原始尺度上的整体误差
mse_original = np.mean((y_pred_original - y_test_original) ** 2)
mae_original = np.mean(np.abs(y_pred_original - y_test_original))

print(f"📋 Overall Performance Summary:")
print(f"   • MSE (original scale): {mse_original:.6f}")
print(f"   • MAE (original scale): {mae_original:.6f}")
print(f"   • Test Loss (scaled): {test_loss:.6f}")
print(f"   • Test MAE (scaled): {test_mae:.6f}")
print()

# ========================
# 7. 保存模型和结果
# ========================
print("💾 Saving model and results...")
model_path = "dnn_model_aerosol.pth"
model.save_model(model_path)

# 保存训练历史
import json
with open("training_history_aerosol.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"✅ Files saved:")
print(f"   • Model: {model_path}")
print(f"   • Feature scaler: scaler_features_aerosol.pkl")
print(f"   • Target scaler: scaler_y_aerosol.pkl")
print(f"   • Training history: training_history_aerosol.json")

print("\n🎉 Training pipeline completed successfully!")
print("=" * 80)