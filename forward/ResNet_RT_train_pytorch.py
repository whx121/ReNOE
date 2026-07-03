import os
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import time
from pathlib import Path
import warnings


from ResNet_RTModel_pytorch import DNNModel, count_parameters

# 设置matplotlib中文字体（可选）
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
warnings.filterwarnings('ignore')

# 创建结果保存目录
def create_directories():
    """创建必要的目录"""
    script_dir = Path(__file__).resolve().parent

    dirs_to_create = ['models', 'scalers', 'plots', 'results']
    created_dirs_paths = []

    for dir_name in dirs_to_create:
        # 构建完整的目标目录路径
        target_dir = script_dir / dir_name
        
        # 创建目录
        target_dir.mkdir(exist_ok=True)
        created_dirs_paths.append(target_dir)
        print(f"创建了目录: {target_dir}") # 方便查看
        
    return created_dirs_paths

class MultiWavelengthTrainer:
    """多波长模型训练器"""
    
    def __init__(self, data_path: str, model_config: Dict = None):
        self.data_path = data_path
        self.model_config = model_config or {
            'layers_config': [1024, 512, 256],
            'learning_rate': 0.001,
            'dropout_rate': 0.001,
            'l2_regularization': 0.001
        }
        
        # 波长和偏振配置
        self.wavelengths = ["0.443", "0.49", "0.565", "0.67", "0.865", "1.02"]
        self.polarization_flag = [0, 1, 0, 1, 1, 0]  # 0 表示无偏振，1 表示有偏振
        
        # 特征列
        self.feature_cols = [
            "sza", "vza", "fis", 
            "vc_BB", "vc_Urban", "vc_Ocean", "vc_Dust", "ALH", 
            "BRDF1", "k1", "k2", "BPDF",
            'o3', 'h2o', 'dem'
        ]
        
        # 存储训练结果
        self.training_results = {}
        self.models = {}
        self.scalers = {}
        
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """加载和准备数据"""
        print("Loading data from parquet file...")
        start_time = time.time()
        
        # 读取数据
        data = pq.read_table(self.data_path).to_pandas()
        print(f"Data loaded in {time.time() - start_time:.2f} seconds")
        print(f"Data shape: {data.shape}")
        
        # 准备特征数据
        features = data[self.feature_cols].values
        print(f"Features shape: {features.shape}")
        
        # 检查缺失值
        missing_features = np.isnan(features).sum()
        if missing_features > 0:
            print(f"Warning: Found {missing_features} missing values in features")
            # 简单处理：用均值填充
            features = np.nan_to_num(features, nan=np.nanmean(features, axis=0))
        
        # 特征标准化
        print("Standardizing features...")
        scaler_features = StandardScaler()
        features_scaled = scaler_features.fit_transform(features)
        
        # 保存特征scaler
        joblib.dump(scaler_features, "/media/amers/SSD_part1/whx/ResNet_code/forward/scalers/scaler_features.pkl")
        print("Features StandardScaler saved to scalers/scaler_features.pkl")

        # 准备目标变量
        targets_dict = {}
        for i, wl in enumerate(self.wavelengths):
            if self.polarization_flag[i] == 1:
                targets_dict[wl] = [f"{wl}_Reflectance", f"{wl}_DOLP"]
            else:
                targets_dict[wl] = [f"{wl}_Reflectance"]
        
        # 提取所有目标变量数据
        all_target_cols = [col for sublist in targets_dict.values() for col in sublist]
        
        # 检查目标变量是否存在
        missing_cols = [col for col in all_target_cols if col not in data.columns]
        if missing_cols:
            print(f"Warning: Missing target columns: {missing_cols}")
            # 移除缺失的列
            all_target_cols = [col for col in all_target_cols if col in data.columns]
        
        y_all = data[all_target_cols].values
        print(f"Targets shape: {y_all.shape}")
        
        # 检查目标变量的缺失值
        missing_targets = np.isnan(y_all).sum()
        if missing_targets > 0:
            print(f"Warning: Found {missing_targets} missing values in targets")
            # 移除包含缺失值的行
            valid_rows = ~np.isnan(y_all).any(axis=1) & ~np.isnan(features_scaled).any(axis=1)
            features_scaled = features_scaled[valid_rows]
            y_all = y_all[valid_rows]
            print(f"After removing missing values - Features: {features_scaled.shape}, Targets: {y_all.shape}")
        
        return features_scaled, y_all, targets_dict
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, val_size: float = 0.2, 
                   random_state: int = 42) -> Tuple:
        """划分数据集"""
        print("Splitting data...")
        
        # 第一次划分：分离测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 第二次划分：从剩余数据中分离验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state + 1
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_target_indices(self, targets_dict: Dict) -> Dict:
        """计算目标变量索引映射"""
        target_indices = {}
        start_idx = 0
        for wl, cols in targets_dict.items():
            num_targets = len(cols)
            target_indices[wl] = (start_idx, start_idx + num_targets)
            start_idx += num_targets
        return target_indices
    
    def train_single_wavelength(self, wl: str, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                               y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                               epochs: int = 1000, batch_size: int = 1000) -> Dict:
        """训练单个波长的模型"""
        print(f"\n{'='*60}")
        print(f"Training model for wavelength: {wl}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # 标准化目标变量
        print("Standardizing target variables...")
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)
        y_test_scaled = y_scaler.transform(y_test)
        
        # 保存目标变量scaler
        scaler_y_filename = f"/media/amers/SSD_part1/whx/ResNet_code/forward/scalers/scaler_y_{wl}.pkl"
        joblib.dump(y_scaler, scaler_y_filename)
        print(f"Target StandardScaler saved to {scaler_y_filename}")
        self.scalers[wl] = y_scaler
        
        # 打印数据统计信息
        print(f"Target shape: {y_train.shape}")
        print(f"Target mean (original): {y_train.mean(axis=0)}")
        print(f"Target std (original): {y_train.std(axis=0)}")
        
        # 初始化模型
        model = DNNModel(
            input_dim=X_train.shape[1],
            output_dim=y_train.shape[1],
            **self.model_config
        )
        
        # 打印模型信息
        print(f"\nModel Architecture for {wl}:")
        print(f"Input dimension: {X_train.shape[1]}")
        print(f"Output dimension: {y_train.shape[1]}")
        print(f"Hidden layers: {self.model_config['layers_config']}")
        count_parameters(model)
        
        # 训练模型
        print(f"\nStarting training...")
        history = model.train_model(
            X_train, y_train_scaled, 
            X_val, y_val_scaled,
            epochs=epochs, 
            batch_size=batch_size
        )
        
        # 评估模型
        print(f"\nEvaluating model on test set...")
        test_loss, test_mae = model.evaluate(X_test, y_test_scaled)
        
        # 在原始尺度上评估
        test_predictions_scaled = model.predict(X_test)
        test_predictions = y_scaler.inverse_transform(test_predictions_scaled)
        
        # 计算原始尺度上的指标
        original_mae = np.mean(np.abs(test_predictions - y_test))
        original_rmse = np.sqrt(np.mean((test_predictions - y_test) ** 2))
        
        print(f"Test MAE (original scale): {original_mae:.6f}")
        print(f"Test RMSE (original scale): {original_rmse:.6f}")
        
        # 保存模型
        model_path = f"/media/amers/SSD_part1/whx/ResNet_code/forward/models/dnn_model_{wl}.pth"
        model.save_model(model_path)
        self.models[wl] = model
        
        # 绘制训练历史
        plot_path = f"/media/amers/SSD_part1/whx/ResNet_code/forward/plots/training_history_{wl}.png"
        model.plot_training_history(save_path=plot_path)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # 返回训练结果
        results = {
            'wavelength': wl,
            'test_loss_scaled': test_loss,
            'test_mae_scaled': test_mae,
            'test_mae_original': original_mae,
            'test_rmse_original': original_rmse,
            'training_time': training_time,
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'epochs_trained': len(history['train_loss']),
            'model_path': model_path,
            'scaler_path': scaler_y_filename
        }
        
        return results
    
    def train_all_wavelengths(self, epochs: int = 1000, batch_size: int = 1000):
        """训练所有波长的模型"""
        print("Starting multi-wavelength training...")
        
        # 创建目录
        create_directories()
        
        # 加载和准备数据
        features_scaled, y_all, targets_dict = self.load_and_prepare_data()
        
        # 划分数据
        X_train, X_val, X_test, y_train_all, y_val_all, y_test_all = self.split_data(
            features_scaled, y_all
        )
        
        # 获取目标变量索引
        target_indices = self.get_target_indices(targets_dict)
        
        # 逐波长训练
        for wl, (start, end) in target_indices.items():
            print(f"\nProcessing wavelength {wl} (indices {start}:{end})")
            
            # 提取当前波长的目标数据
            y_train = y_train_all[:, start:end]
            y_val = y_val_all[:, start:end]
            y_test = y_test_all[:, start:end]
            
            # 训练模型
            results = self.train_single_wavelength(
                wl, X_train, X_val, X_test, y_train, y_val, y_test,
                epochs=epochs, batch_size=batch_size
            )
            
            self.training_results[wl] = results
        
        # 保存训练总结
        self.save_training_summary()
        
        # 绘制总结图表
        self.plot_training_summary()
        
        print("\n" + "="*60)
        print("All wavelengths training completed!")
        print("="*60)
    
    def save_training_summary(self):
        """保存训练总结"""
        summary_df = pd.DataFrame(self.training_results).T
        summary_path = "/media/amers/SSD_part1/whx/ResNet_code/forward/results/training_summary.csv"
        summary_df.to_csv(summary_path)
        print(f"Training summary saved to {summary_path}")
        
        # 保存详细的配置信息
        config_info = {
            'model_config': self.model_config,
            'wavelengths': self.wavelengths,
            'polarization_flag': self.polarization_flag,
            'feature_cols': self.feature_cols,
            'data_path': self.data_path
        }
        
        import json
        with open("/media/amers/SSD_part1/whx/ResNet_code/forward/results/training_config.json", 'w') as f:
            json.dump(config_info, f, indent=2)
        print("Training configuration saved to results/training_config.json")
    
    def plot_training_summary(self):
        """绘制训练总结图表"""
        if not self.training_results:
            print("No training results to plot.")
            return
        
        # 创建总结图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        wavelengths = list(self.training_results.keys())
        
        # 1. 测试MAE对比
        mae_original = [self.training_results[wl]['test_mae_original'] for wl in wavelengths]
        mae_scaled = [self.training_results[wl]['test_mae_scaled'] for wl in wavelengths]
        
        x = np.arange(len(wavelengths))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, mae_original, width, label='Original Scale', alpha=0.8)
        axes[0, 0].bar(x + width/2, mae_scaled, width, label='Scaled', alpha=0.8)
        axes[0, 0].set_xlabel('Wavelength')
        axes[0, 0].set_ylabel('Test MAE')
        axes[0, 0].set_title('Test MAE by Wavelength')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(wavelengths, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 训练时间
        training_times = [self.training_results[wl]['training_time'] for wl in wavelengths]
        axes[0, 1].bar(wavelengths, training_times, alpha=0.8, color='orange')
        axes[0, 1].set_xlabel('Wavelength')
        axes[0, 1].set_ylabel('Training Time (seconds)')
        axes[0, 1].set_title('Training Time by Wavelength')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 训练轮数
        epochs_trained = [self.training_results[wl]['epochs_trained'] for wl in wavelengths]
        axes[1, 0].bar(wavelengths, epochs_trained, alpha=0.8, color='green')
        axes[1, 0].set_xlabel('Wavelength')
        axes[1, 0].set_ylabel('Epochs Trained')
        axes[1, 0].set_title('Epochs Trained by Wavelength')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. RMSE
        rmse_values = [self.training_results[wl]['test_rmse_original'] for wl in wavelengths]
        axes[1, 1].bar(wavelengths, rmse_values, alpha=0.8, color='red')
        axes[1, 1].set_xlabel('Wavelength')
        axes[1, 1].set_ylabel('Test RMSE (Original Scale)')
        axes[1, 1].set_title('Test RMSE by Wavelength')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        #plt.savefig("forward/plots/training_summary.png", dpi=300, bbox_inches='tight')
        #plt.show()
        
        print("Training summary plots saved to plots/training_summary.png")

def main():
    """主函数"""
    # 配置参数
    DATA_PATH = '/media/amers/WHX/polder_simulation_results/train_data_for_polder.parquet'
    
    # 模型配置
    MODEL_CONFIG = {
        'layers_config': [1024, 256, 128],
        'learning_rate': 0.001,
        'dropout_rate': 0.001,
        'l2_regularization': 0.0001
    }
    
    # 训练配置
    TRAINING_CONFIG = {
        'epochs': 1000,
        'batch_size': 2048
    }
    
    # 检查数据文件是否存在
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please update the DATA_PATH variable with the correct path.")
        return
    
    # 创建训练器
    trainer = MultiWavelengthTrainer(
        data_path=DATA_PATH,
        model_config=MODEL_CONFIG
    )
    
    # 开始训练
    try:
        trainer.train_all_wavelengths(**TRAINING_CONFIG)
        
        # 打印最终结果摘要
        print("\nFinal Results Summary:")
        print("-" * 50)
        for wl, results in trainer.training_results.items():
            print(f"Wavelength {wl}:")
            print(f"  Test MAE: {results['test_mae_original']:.6f}")
            print(f"  Test RMSE: {results['test_rmse_original']:.6f}")
            print(f"  Training time: {results['training_time']:.2f}s")
            print(f"  Epochs: {results['epochs_trained']}")
            print()
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()