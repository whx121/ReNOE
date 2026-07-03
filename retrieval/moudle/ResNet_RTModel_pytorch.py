import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from typing import List, Tuple, Optional
import pickle
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm import tqdm
# 设置随机种子以确保可重现性
torch.manual_seed(42)
np.random.seed(42)

class ResidualBlock(nn.Module):
    """残差块实现"""
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.2, 
                 l2_reg: float = 0.01):
        super(ResidualBlock, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 主路径
        self.main_path = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate)
        )
        
        # 跳跃连接：如果输入输出维度不同，需要投影
        self.shortcut = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        
        # 应用权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.main_path(x)
        return out + identity

class DNNModel(nn.Module):
    """
    改进的深度神经网络模型，包含残差连接和层归一化
    """
    def __init__(self, input_dim: int, output_dim: int, 
                 layers_config: List[int] = [128, 64, 32],
                 learning_rate: float = 0.001, 
                 dropout_rate: float = 0.2, 
                 l2_regularization: float = 0.01,
                 device: str = 'auto'):
        super(DNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_config = layers_config
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.l2_regularization = l2_regularization
        
        # 自动选择设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # 构建网络
        self.network = self._build_network()
        self.to(self.device)
        
        # 初始化优化器和损失函数
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.scheduler = None
        
        # 训练历史
        self.train_history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    
    def _build_network(self):
        """构建网络结构"""
        layers = nn.ModuleList()
        
        # 输入层
        current_dim = self.input_dim
        
        # 隐藏层（使用残差连接）
        for i, units in enumerate(self.layers_config):
            if i == 0:
                # 第一层直接连接
                layers.append(nn.Sequential(
                    nn.Linear(current_dim, units),
                    nn.LayerNorm(units),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Dropout(self.dropout_rate)
                ))
            else:
                # 后续层使用残差连接
                layers.append(ResidualBlock(current_dim, units, self.dropout_rate, self.l2_regularization))
            current_dim = units
        
        # 输出层
        layers.append(nn.Linear(current_dim, self.output_dim))
        
        return layers
    
    def forward(self, x):
        """前向传播"""
        for layer in self.network:
            x = layer(x)
        return x
    
    def _setup_training(self):
        """设置训练相关组件"""
        # 优化器（带权重衰减实现L2正则化）
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_regularization
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.2,
            patience=50,
            min_lr=1e-9,
            # verbose=True
        )
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                epochs: int = 100, batch_size: int = 32,
                early_stopping_patience: int = 150) -> dict:
        """
        训练模型
        
        Args:
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            epochs: 训练轮数
            batch_size: 批大小
            early_stopping_patience: 早停patience
        
        Returns:
            训练历史字典
        """
        # 设置训练组件
        self._setup_training()
        
        # 转换为张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 早停相关变量
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # 训练阶段
            self.train()
            train_loss = 0.0
            train_mae = 0.0
            
            # 创建batch进度条
            batch_pbar = tqdm(train_loader, 
                            desc=f"Epoch {epoch+1}/{epochs}", 
                            leave=False,
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}] - {postfix}')
            
            for batch_X, batch_y in batch_pbar:
                self.optimizer.zero_grad()
                
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                batch_train_loss = loss.item()
                batch_train_mae = torch.mean(torch.abs(outputs - batch_y)).item()
                
                train_loss += batch_train_loss
                train_mae += batch_train_mae
                
                # 更新batch进度条显示当前batch的指标
                batch_pbar.set_postfix({
                    'loss': f'{batch_train_loss:.4f}',
                    'mae': f'{batch_train_mae:.4f}'
                })
            
            train_loss /= len(train_loader)
            train_mae /= len(train_loader)
            
            # 验证阶段
            self.eval()
            with torch.no_grad():
                val_outputs = self(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
                val_mae = torch.mean(torch.abs(val_outputs - y_val_tensor)).item()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['train_mae'].append(train_mae)
            self.train_history['val_mae'].append(val_mae)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.state_dict().copy()
            else:
                patience_counter += 1
            
            # 每个epoch结束后打印详细信息
            print(f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f} - "
                f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f} - "
                f"LR: {current_lr:.2e} - Patience: {patience_counter}/{early_stopping_patience}")
            
            # 早停
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # 加载最佳模型
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            print(f"Loaded best model with validation loss: {best_val_loss:.6f}")
        
        return self.train_history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """评估模型"""
        self.eval()
        
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        with torch.no_grad():
            predictions = self(X_test_tensor)
            test_loss = self.criterion(predictions, y_test_tensor).item()
            test_mae = torch.mean(torch.abs(predictions - y_test_tensor)).item()
        
        print(f"Test Loss: {test_loss:.6f}, Test MAE: {test_mae:.6f}")
        return test_loss, test_mae
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        self.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self(X_tensor)
        
        return predictions.cpu().numpy()
    
    def compute_jacobian(self, X: np.ndarray) -> np.ndarray:
        """
        计算雅可比矩阵
        
        Args:
            X: 输入数据 (batch_size, input_dim)
        
        Returns:
            雅可比矩阵 (batch_size, output_dim, input_dim)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        X_tensor.requires_grad_(True)
        
        outputs = self(X_tensor)
        
        jacobians = []
        for i in range(outputs.shape[1]):  # 对每个输出维度
            grad_outputs = torch.zeros_like(outputs)
            grad_outputs[:, i] = 1
            
            grads = torch.autograd.grad(
                outputs=outputs,
                inputs=X_tensor,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False
            )[0]
            
            jacobians.append(grads.unsqueeze(1))
        
        jacobian = torch.cat(jacobians, dim=1)
        return jacobian.detach().cpu().numpy()
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'layers_config': self.layers_config,
                'learning_rate': self.learning_rate,
                'dropout_rate': self.dropout_rate,
                'l2_regularization': self.l2_regularization
            },
            'train_history': self.train_history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'auto'):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        config = checkpoint['model_config']
        
        # 创建模型实例
        model = cls(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            layers_config=config['layers_config'],
            learning_rate=config['learning_rate'],
            dropout_rate=config['dropout_rate'],
            l2_regularization=config['l2_regularization'],
            device=device
        )
        
        # 加载状态字典
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载训练历史（如果有）
        if 'train_history' in checkpoint:
            model.train_history = checkpoint['train_history']
        
        #print(f"Model loaded from {filepath}")
        return model
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """绘制训练历史"""
        if not self.train_history['train_loss']:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_history['train_loss'], label='Train Loss', alpha=0.8)
        ax1.plot(self.train_history['val_loss'], label='Validation Loss', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # MAE曲线
        ax2.plot(self.train_history['train_mae'], label='Train MAE', alpha=0.8)
        ax2.plot(self.train_history['val_mae'], label='Validation MAE', alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_title('Training and Validation MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()

# 工具函数
def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params