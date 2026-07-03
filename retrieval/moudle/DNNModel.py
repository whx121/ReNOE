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