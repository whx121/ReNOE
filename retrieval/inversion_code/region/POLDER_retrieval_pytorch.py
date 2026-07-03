import os

import h5py

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1' # 针对macOS上的Accelerate框架
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # 针对HuggingFace Tokenizers

import numpy as np
import pandas as pd
from time import time
import warnings
import logging
from scipy.optimize import least_squares, differential_evolution
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
import torch
import argparse
import sys
import multiprocessing as mp
import subprocess
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import gc

# 设置环境
warnings.filterwarnings('ignore', category=UserWarning)

device = torch.device('cpu')
torch.set_num_threads(1)
# 导入自定义模块

from retrieval.moudle.ResNet_RTModel_pytorch import DNNModel
import retrieval.moudle.moudle_OE_vector as OEfunc
from retrieval.moudle.predict_forAOD import predict_AOD_AAOD_SSA as predict_AOD
from retrieval.moudle.moudle_OE_vector import (
    vectorized_extract_polder_data,
    calculate_row_col
)

@dataclass
class RetrievalConfig:
    """反演配置类，便于管理和修改参数"""

    # 波长设置
    wl_list: List[str] = field(default_factory=lambda: ["0.443", "0.49", "0.565", "0.67", "0.865", "1.02"])
    has_polarization: Dict[str, bool] = field(default_factory=lambda: {
        "0.443": False, "0.49": True, "0.565": False,
        "0.67": True, "0.865": True, "1.02": False
    })

    # 参数定义
    total_parameters: List[str] = field(default_factory=lambda: [
        "sza", "vza", "fis", "vc_BB", "vc_Urban", "vc_Ocean", "vc_Dust", "ALH",
        "iso_0.443", "iso_0.49", "iso_0.565", "iso_0.67", "iso_0.865", "iso_1.02",
        "k1", "k2", "BPDF", "o3", "h2o", "dem"
    ])

    # 状态向量（可反演参数）
    state_vector_list: List[str] = field(default_factory=lambda: [
        "vc_BB", "vc_Urban", "vc_Ocean", "vc_Dust", "ALH",
        "iso_0.443", "iso_0.49", "iso_0.565", "iso_0.67", "iso_0.865", "iso_1.02",
        "k1", "k2", "BPDF"
    ])

    # 非状态向量（固定参数）
    nonstate_vector_list: List[str] = field(default_factory=lambda: [
        "sza", "vza", "fis", "o3", "h2o", "dem"
    ])

    # 误差设置
    sigma_I: float = 0.03  # 相对测量误差
    sigma_dolp: float = 0.01  # 绝对测量误差
    sigma_NN_I: List[float] = field(default_factory=lambda: [0.0006, 0.0007, 0.0007, 0.0007, 0.0007, 0.0005])
    sigma_NN_DOLP: List[float] = field(default_factory=lambda: [0.0, 0.0008, 0.0, 0.0008, 0.0008, 0.0])

    # 优化设置
    optimization_method: str = 'trf'  # 'trf', 'dogbox', 'lm'
    max_iterations: int = 15  # 最大迭代次数
    xtol: float = 1e-7  # 变量参数容差
    ftol: float = 1e-7  # 函数容差
    gtol: float = 1e-7  # 梯度容差

    # 正则化设置
    use_regularization: bool = True
    regularization_weight: float = 0.005

    # 先验设置
    use_prior: bool = True
    prior_weight: float = 0.5

    # 先验类型：区分真实先验和初始猜测
    prior_type: Dict[str, str] = field(default_factory=lambda: {
        "vc_BB": "guess",  # 初始猜测，非真实先验
        "vc_Urban": "guess",  # 初始猜测
        "vc_Ocean": "guess",  # 初始猜测
        "vc_Dust": "guess",  # 初始猜测
        "ALH": "model",
        "iso": "climatology",
        "k1": "climatology",
        "k2": "climatology",
        "BPDF": "model"
    })

    # 先验不确定性（sigma）：根据数据来源设置
    prior_sigma: Dict[str, float] = field(default_factory=lambda: {
        # 气溶胶成分：无真实先验，设置很大的不确定性
        "vc_BB": 0.5,  # 极大不确定性 = 几乎不约束
        "vc_Urban": 0.5,  # 极大不确定性
        "vc_Ocean": 0.5,  # 极大不确定性
        "vc_Dust": 0.5,  # 极大不确定性

        # 气溶胶层高度：气候学数据，中等不确定性
        "ALH": 0.5,  # 1km标准差，中等约束

        # 地表反照率：MODIS产品，较小不确定性
        "iso": 0.1,  # 3%绝对误差，较强约束

        # BRDF参数：取决于数据质量
        "k1": 0.1,  #
        "k2": 0.1,  #

        # 偏振BRDF：统计数据，中等不确定性
        "BPDF": 0.3  # 中等约束
    })

    # 根据先验类型自动调整权重
    prior_weight_factor: Dict[str, float] = field(default_factory=lambda: {
        "observation": 1.0,  # 观测数据：正常权重
        "climatology": 1.0,  # 气候学数据：降低权重
        "guess": 0.000000000001,  # 初始猜测：极低权重
        "model": 1.0  # 模型数据：中等权重
    })

    def __post_init__(self):
        """初始化后处理"""
        self.K = len(self.state_vector_list)
        self._init_bounds()
        self._init_state()
        self._init_obs_count()

    def _init_bounds(self):
        """初始化参数边界"""
        self.state_bounds = [
            (0, 1),  # vc_BB
            (0, 1),  # vc_Urban
            (0, 1),  # vc_Ocean
            (0.00001, 1),  # vc_Dust
            (0.5, 8),  # ALH
            (0.001, 0.8),  # iso_0.443
            (0.001, 0.8),  # iso_0.490
            (0.001, 0.8),  # iso_0.565
            (0.001, 0.8),  # iso_0.67
            (0.005, 0.8),  # iso_0.865
            (0.005, 0.8),  # iso_1.02
            (0.005, 2),  # k1
            (0.005, 2),  # k2
            (0.5, 8)  # BPDF
        ]

    def _init_state(self):
        """初始化状态向量"""
        self.init_state = np.array([
            0.1,  # vc_BB
            0.1,  # vc_Urban
            0.1,  # vc_Ocean
            0.2,  # vc_Dust
            3.5,  # ALH
            0.05,  # iso_0.443
            0.06,  # iso_0.490
            0.1,  # iso_0.565
            0.12,  # iso_0.67
            0.3,  # iso_0.865
            0.4,  # iso_1.02
            0.6,  # k1
            0.4,  # k2
            4  # BPDF
        ])

    def _init_obs_count(self):
        """初始化观测数量"""
        self.obs_count_per_wl = {
            wl: 2 if self.has_polarization[wl] else 1
            for wl in self.wl_list
        }

    def save_config(self, filepath: str):
        """保存配置到JSON文件"""
        config_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
        # 转换numpy数组为列表
        for key, value in config_dict.items():
            if isinstance(value, np.ndarray):
                config_dict[key] = value.tolist()

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_config(cls, filepath: str):
        """从JSON文件加载配置"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        # 转换列表为numpy数组
        if 'init_state' in config_dict:
            config_dict['init_state'] = np.array(config_dict['init_state'])

        return cls(**config_dict)


class ImprovedOptimizer:
    """改进的优化器类，提供多种优化策略"""

    def __init__(self, config: RetrievalConfig):
        self.config = config

    def optimize_with_multistart(self, data, r_obs, model_dict, features_scaler,
                                 n_starts: int = 3) -> Tuple[np.ndarray, float]:
        """多起点优化策略"""
        best_state = None
        best_cost = np.inf

        for i in range(n_starts):
            # 生成不同的初始点
            if i == 0:
                init_state = self.config.init_state
            else:
                # 在边界内随机生成初始点
                init_state = self._generate_random_state()

            # 执行优化
            result = self._optimize_single(data, r_obs, model_dict, features_scaler, init_state)

            # 评估结果
            final_cost, _, _ = current_total_cost_function_multi(
                result.x, data, r_obs, model_dict, features_scaler, self.config
            )

            if final_cost < best_cost:
                best_cost = final_cost
                best_state = result.x

        return best_state, best_cost

    def optimize_with_adaptive_weights(self, data, r_obs, model_dict, features_scaler,
                                       n_starts: int = 2) -> Tuple[np.ndarray, float, Dict]:
        """自适应权重的优化策略"""
        import copy

        # 创建配置副本，避免修改原始配置
        adaptive_config = copy.deepcopy(self.config)

        # 评估观测质量
        n_angles = r_obs.shape[0]
        n_obs_per_angle = r_obs.shape[1]
        has_pol = n_obs_per_angle > len(self.config.wl_list)  # 有偏振

        # 自适应调整权重
        if n_angles < 5:
            # 观测少，增加约束
            adaptive_config.prior_weight *= 1.5
            adaptive_config.regularization_weight *= 1.5
            # print(f"观测角度少({n_angles})，增加约束权重")
        elif n_angles > 10 and has_pol:
            # 观测充足且有偏振，降低约束
            adaptive_config.prior_weight *= 1
            adaptive_config.regularization_weight *= 1
            # print(f"观测充足({n_angles}角度+偏振)，降低约束权重")

        # 使用调整后的配置创建新的优化器
        adaptive_optimizer = ImprovedOptimizer(adaptive_config)

        # 执行优化
        best_state, best_cost = adaptive_optimizer.optimize_with_multistart(
            data, r_obs, model_dict, features_scaler, n_starts
        )

        # 诊断信息
        diagnostics = {
            'n_angles': n_angles,
            'has_polarization': has_pol,
            'adapted_prior_weight': adaptive_config.prior_weight,
            'adapted_reg_weight': adaptive_config.regularization_weight
        }

        return best_state, best_cost, diagnostics

    def _generate_random_state(self) -> np.ndarray:
        """在边界内生成随机状态向量"""
        state = np.zeros(self.config.K)
        for i, (low, high) in enumerate(self.config.state_bounds):
            # 使用对数均匀分布，对小值参数更友好
            if low > 0 and high / low < 1000:  # 避免数值问题
                state[i] = np.exp(np.random.uniform(np.log(low), np.log(high)))
            else:
                state[i] = np.random.uniform(low, high)
        return state

    def check_feasibility(self, state_vector: np.ndarray) -> Tuple[bool, List[str]]:
        """检查状态向量是否在边界内"""
        is_feasible = True
        violations = []

        for i, (value, (low, high)) in enumerate(zip(state_vector, self.config.state_bounds)):
            if value < low or value > high:
                is_feasible = False
                param_name = self.config.state_vector_list[i]
                violations.append(f"{param_name}: {value:.4f} not in [{low:.4f}, {high:.4f}]")

        return is_feasible, violations

    def _optimize_single(self, data, r_obs, model_dict, features_scaler,
                         init_state) -> object:
        """单次优化"""
        # 检查初始值可行性
        is_feasible, violations = self.check_feasibility(init_state)
        if not is_feasible:
            print(f"警告：初始值不可行！违反约束：{violations}")
            # 尝试修复初始值
            init_state = np.array([
                np.clip(init_state[i], self.config.state_bounds[i][0], self.config.state_bounds[i][1])
                for i in range(len(init_state))
            ])
            print("已将初始值裁剪到边界内")

        n_total = r_obs.size

        # 计算权重矩阵
        weights_np = compute_weights_improved(r_obs, self.config)
        sqrt_weights = np.sqrt(weights_np)

        # 获取先验信息
        prior_state = init_state.copy()
        prior_weights = self._get_prior_weights()

        def residual_function(sv):
            _, r_sim, _ = current_total_cost_function_multi(
                sv, data, r_obs, model_dict, features_scaler, self.config
            )
            diff = r_obs - r_sim
            residuals = (1.0 / np.sqrt(n_total)) * (sqrt_weights * diff).ravel()

            # 添加先验约束
            if self.config.use_prior:
                prior_residuals = self.config.prior_weight * prior_weights * (sv - prior_state)

                residuals = np.concatenate([residuals, prior_residuals])

            # 添加正则化项
            if self.config.use_regularization:
                reg_residuals = self.config.regularization_weight * sv
                residuals = np.concatenate([residuals, reg_residuals])

            return residuals

        def jac_function(sv):
            _, _, jacobian = current_total_cost_function_multi(
                sv, data, r_obs, model_dict, features_scaler, self.config
            )
            J_val = -(sqrt_weights / np.sqrt(n_total)).reshape(-1, 1) * \
                    jacobian.reshape(-1, jacobian.shape[-1])

            # 添加先验雅可比
            if self.config.use_prior:
                prior_jac = self.config.prior_weight * np.diag(prior_weights)
                J_val = np.vstack([J_val, prior_jac])

            # 添加正则化雅可比
            if self.config.use_regularization:
                reg_jac = self.config.regularization_weight * np.eye(self.config.K)
                J_val = np.vstack([J_val, reg_jac])

            return J_val

        try:
            # print(self.config.state_bounds)
            result = least_squares(
                fun=residual_function,
                jac=jac_function,
                x0=init_state,
                bounds=([b[0] for b in self.config.state_bounds],
                        [b[1] for b in self.config.state_bounds]),
                method=self.config.optimization_method,
                xtol=self.config.xtol,
                ftol=self.config.ftol,
                gtol=self.config.gtol,
                max_nfev=self.config.max_iterations,
                x_scale='jac',
                verbose=0
            )
        except Exception as e:
            print(f"优化失败: {e}")

            # 返回一个失败的结果对象
            class FailedResult:
                def __init__(self, x):
                    self.x = x
                    self.success = False
                    self.message = str(e)

            result = FailedResult(init_state)

        return result

    def _get_prior_weights(self) -> np.ndarray:
        """获取先验权重，考虑先验类型"""
        weights = np.zeros(self.config.K)

        for i, param in enumerate(self.config.state_vector_list):
            # 基础权重 = 1/sigma
            if param.startswith('vc_'):
                base_weight = 1.0 / self.config.prior_sigma.get('vc_BB', 1.0)
            elif param.startswith('iso_'):
                base_weight = 1.0 / self.config.prior_sigma.get('iso', 0.1)
            elif param in self.config.prior_sigma:
                base_weight = 1.0 / self.config.prior_sigma[param]
            else:
                base_weight = 1.0

            # 根据先验类型调整权重
            param_base = param.split('_')[0] if '_' in param else param
            prior_type = self.config.prior_type.get(param_base, 'guess')
            type_factor = self.config.prior_weight_factor.get(prior_type, 0.01)

            weights[i] = base_weight * type_factor

        return weights


def compute_weights_improved(r_obs: np.ndarray, config: RetrievalConfig) -> np.ndarray:
    """改进的权重计算，考虑观测值大小和测量误差"""
    weights = np.zeros_like(r_obs)
    current_col = 0

    for i, wl in enumerate(config.wl_list):
        # 反射率权重
        I_obs = r_obs[:, current_col]
        sigma_NN_I = config.sigma_NN_I[i]

        # 动态误差模型：考虑相对误差和绝对误差
        sigma_I_dynamic = np.sqrt((config.sigma_I * I_obs) ** 2 + sigma_NN_I ** 2)

        # 避免过小的权重
        min_sigma = 0.0001
        sigma_I_dynamic = np.maximum(sigma_I_dynamic, min_sigma)

        weights[:, current_col] = 1.0 / (sigma_I_dynamic ** 2)
        current_col += 1

        # DOLP权重
        if config.has_polarization[wl]:
            sigma_NN_DOLP = config.sigma_NN_DOLP[i]
            if sigma_NN_DOLP > 0:  # 只有当有偏振测量误差时才计算
                sigma_DOLP_dynamic = np.sqrt(config.sigma_dolp ** 2 + sigma_NN_DOLP ** 2)
                weights[:, current_col] = 1.0 / (sigma_DOLP_dynamic ** 2)
            else:
                weights[:, current_col] = 0  # 不使用该观测
            current_col += 1

    return weights


def process_multi_angle_data_improved(data_obs, elev, lon, lat, prior_path, config: RetrievalConfig,
                                      priori_flag: bool) -> Tuple[np.ndarray, np.ndarray, RetrievalConfig]:
    """改进的多角度数据处理"""
    n_valid = data_obs.shape[0]

    # 初始化矩阵
    n_obs_cols = sum(config.obs_count_per_wl.values())
    r_obs_matrix = np.zeros((n_valid, n_obs_cols))
    n_non_state = len(config.nonstate_vector_list)
    non_state_matrix = np.zeros((n_valid, n_non_state))

    # 提取观测几何
    non_state_matrix[:, 0:3] = data_obs[:, 0:3]

    # 提取观测值
    current_col_obs = 0
    current_col_data = 3

    for wl in config.wl_list:
        r_obs_matrix[:, current_col_obs] = data_obs[:, current_col_data]
        current_col_obs += 1
        current_col_data += 1

        if config.has_polarization[wl]:
            r_obs_matrix[:, current_col_obs] = data_obs[:, current_col_data]
            current_col_obs += 1
            current_col_data += 1

    # 获取先验数据
    try:
        iso_443, iso_490, iso_565, iso_670, iso_865, iso_1020, k1, k2, BPDF, ALH = \
            OEfunc.extract_prior_data(lon, lat, prior_path)

        # 检查先验数据的合理性
        prior_values = {
            'iso_0.443': np.clip(iso_443, 0.005, 0.5),
            'iso_0.49': np.clip(iso_490, 0.005, 0.5),
            'iso_0.565': np.clip(iso_565, 0.005, 0.6),
            'iso_0.67': np.clip(iso_670, 0.005, 0.6),
            'iso_0.865': np.clip(iso_865, 0.005, 0.8),
            'iso_1.02': np.clip(iso_1020, 0.005, 0.8),
            'k1': np.clip(k1, 0.01, 2),
            'k2': np.clip(k2, 0.01, 2),
            'BPDF': np.clip(BPDF, 0.5, 8),
            'ALH': np.clip(ALH, 0.5, 8)
        }
    except Exception as e:
        print(f"先验数据获取失败: {e}，使用默认值")
        prior_values = None

    # 设置非状态参量
    for i, param in enumerate(config.nonstate_vector_list[3:], 3):
        if param == 'dem':
            non_state_matrix[:, i] = elev
        elif param == 'o3':
            non_state_matrix[:, i] = 340.5  # 可以改为从外部数据源获取
        elif param == 'h2o':
            non_state_matrix[:, i] = 1  # 可以改为从外部数据源获取
        elif param == 'k1' and prior_values:
            non_state_matrix[:, i] = prior_values.get('k1', k1)
        elif param == 'k2' and prior_values:
            non_state_matrix[:, i] = prior_values.get('k2', k2)
        elif param == 'BPDF' and prior_values:
            non_state_matrix[:, i] = prior_values.get('BPDF', BPDF)

    # 如果使用先验，更新初始状态和边界
    if priori_flag and config.use_prior and prior_values:
        new_config = update_config_with_prior(config, prior_values)
        return r_obs_matrix, non_state_matrix, new_config

    return r_obs_matrix, non_state_matrix, config


def update_config_with_prior(config: RetrievalConfig, prior_values: Dict[str, float]) -> RetrievalConfig:
    """使用先验值更新配置，确保初始值在边界内"""
    import copy
    new_config = copy.deepcopy(config)

    for i, param in enumerate(new_config.state_vector_list):
        if param in prior_values:
            value = prior_values[param]

            # 更新边界：以先验值为中心的合理范围
            if param.startswith('iso_'):
                margin = 0.06
            elif param in ['k1', 'k2']:
                margin = 0.1
            elif param == 'BPDF':
                margin = 0.5
            elif param == 'ALH':
                margin = 1
            else:
                margin = 0.4

            # 计算新边界，确保合理
            original_lower, original_upper = new_config.state_bounds[i]
            lower = max(value - margin, original_lower)
            upper = min(value + margin, original_upper)

            # 确保边界有效（lower < upper）
            if lower >= upper:
                # 如果边界无效，使用原始边界
                lower, upper = original_lower, original_upper

            # 确保初始值在边界内
            init_value = np.clip(value, lower, upper)

            new_config.init_state[i] = init_value
            new_config.state_bounds[i] = (lower, upper)

    # 最终检查：确保所有初始值都在边界内
    for i in range(len(new_config.init_state)):
        lower, upper = new_config.state_bounds[i]
        if new_config.init_state[i] < lower or new_config.init_state[i] > upper:
            new_config.init_state[i] = np.clip(new_config.init_state[i], lower, upper)
            print(f"警告：参数 {new_config.state_vector_list[i]} 的初始值已被裁剪到边界内")

    return new_config


def quality_check(optimized_state: np.ndarray, final_cost: float,
                  config: RetrievalConfig) -> Dict[str, any]:
    """质量检查和诊断"""
    quality = {
        'converged': final_cost < 0.05,  # 成本函数阈值
        'final_cost': final_cost,
        'vc_total': None,
        'quality_flag': 0,  # 0: 好, 1: 一般, 2: 差
        'warnings': []
    }

    # 计算AOD（简化为体积浓度总和）
    vc_total = sum(optimized_state[i] for i in range(4))  # 前4个是气溶胶成分
    quality['vc_total'] = vc_total

    # 质量标记
    if final_cost > 0.1:
        quality['quality_flag'] = 2
        quality['warnings'].append('High cost function value')
    elif final_cost > 0.001:
        quality['quality_flag'] = 1
        quality['warnings'].append('Moderate cost function value')

    # 检查参数合理性
    if quality['vc_total'] > 3.0:
        quality['warnings'].append('Unusually high AOD')
        quality['quality_flag'] = max(quality['quality_flag'], 1)

    # 检查气溶胶层高度
    alh_idx = config.state_vector_list.index('ALH')
    if optimized_state[alh_idx] > 10.0:
        quality['warnings'].append('High aerosol layer height')

    return quality


def process_single_pixel_improved(pixel_index: int, data_obs: np.ndarray,
                                  elev: float, lon: float, lat: float, prior_path,
                                  config: RetrievalConfig, model_dict: Dict,
                                  features_scaler, priori_flag: bool) -> Dict:
    """改进的单像元处理函数"""
    try:
        # 处理观测数据
        r_obs_matrix, non_state_matrix, new_config = process_multi_angle_data_improved(
            data_obs, elev, lon, lat, prior_path, config, priori_flag
        )

        if r_obs_matrix.shape[0] == 0:
            return {'pixel_index': pixel_index, 'error': '没有有效观测数据'}

        # 构造输入数据矩阵
        n_rows = non_state_matrix.shape[0]
        total_length = 3 + len(new_config.init_state) + (non_state_matrix.shape[1] - 3)
        data_matrix = np.zeros((n_rows, total_length))
        data_matrix[:, :3] = non_state_matrix[:, :3]
        data_matrix[:, 3:3 + len(new_config.init_state)] = np.tile(new_config.init_state, (n_rows, 1))
        data_matrix[:, 3 + len(new_config.init_state):] = non_state_matrix[:, 3:]

        # 使用改进的优化器（支持自适应权重）
        optimizer = ImprovedOptimizer(new_config)
        optimized_state, final_cost, diagnostics = optimizer.optimize_with_adaptive_weights(
            data_matrix, r_obs_matrix, model_dict, features_scaler, n_starts=1
        )

        # 质量检查
        quality_info = quality_check(optimized_state, final_cost, new_config)
        quality_info['diagnostics'] = diagnostics  # 添加诊断信息

        return {
            'pixel_index': pixel_index,
            'optimized_state': optimized_state,
            'final_cost': final_cost,
            'quality_info': quality_info
        }

    except Exception as e:
        return {
            'pixel_index': pixel_index,
            'error': str(e)
        }


def parallel_pixel_inversion_improved(config: RetrievalConfig, data_obs_list: List,
                                      elev_list: List, lon_list: List, lat_list: List, prior_path,
                                      output_path: str, output_name: str,
                                      model_dict: Dict, features_scaler, polder_time,
                                      priori_flag: bool = True, n_jobs: int = -1) -> List[Dict]:
    """改进的并行反演函数"""
    # start_time = time()
    n_pixels = len(data_obs_list)

    print(f"开始处理 {n_pixels} 个像元...")

    # 并行处理
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_single_pixel_improved)(
            i, data_obs_list[i], elev_list[i], lon_list[i], lat_list[i], prior_path,
            config, model_dict, features_scaler, priori_flag
        )
        for i in tqdm(range(n_pixels), desc="Processing pixels")
    )

    # 处理结果
    all_results = []
    for result in results:
        if 'error' in result:
            result_data = {
                'lon': lon_list[result['pixel_index']],
                'lat': lat_list[result['pixel_index']],
                'final_cost': np.nan,
                'vc_total': np.nan,
                'quality_flag': 3,
                'error': result['error'],
                **{f"{config.state_vector_list[i]}": np.nan for i in range(config.K)}
            }
        else:
            quality_info = result.get('quality_info', {})
            result_data = {
                'time': polder_time,
                'lon': lon_list[result['pixel_index']],
                'lat': lat_list[result['pixel_index']],
                'final_cost': result.get('final_cost', np.nan),
                'vc_total': quality_info.get('vc_total', np.nan),
                'quality_flag': quality_info.get('quality_flag', 3),
                'converged': quality_info.get('converged', False),
                'error': '',
                **{f"{config.state_vector_list[i]}": result.get('optimized_state', [np.nan] * config.K)[i]
                   for i in range(config.K)}
            }
        all_results.append(result_data)

    # 保存结果
    result_df = pd.DataFrame(all_results)
    result_file = os.path.join(output_path, output_name)

    # 保存结果
    if os.path.exists(result_file):
        result_df.to_csv(result_file, mode='a', header=False, index=False)
    else:
        result_df.to_csv(result_file, index=False)

    print(f"结果保存至: {result_file}")

    return all_results


# =============== PYTORCH MODIFICATION START ===============

def declare_multi_model(wl_list: List[str], model_dir: str, device: str = 'cpu') -> Dict:
    """
    加载多波段模型 (PyTorch version)
    """
    model_dict = {}

    for wl in wl_list:
        model_filename = os.path.join(model_dir, f"dnn_model_{wl}.pth")
        scaler_filename = os.path.join(model_dir, f"scaler_y_{wl}.pkl")

        try:
            # 使用自定义的load_model方法
            model = DNNModel.load_model(model_filename, device=device)
            model.eval()

            # 加载对应的目标变量缩放器
            target_scaler = joblib.load(scaler_filename)
            model_dict[wl] = (model, target_scaler)

        except FileNotFoundError:
            print(f"[错误] 找不到PyTorch模型文件: {model_filename}")
            print("请确认模型文件存在且路径正确。")
            raise

    return model_dict


def update_state_in_data(data: np.ndarray, state_vector: np.ndarray,
                         config: RetrievalConfig) -> np.ndarray:
    """向量化更新数据中的状态变量"""
    n = data.shape[0]
    m = len(config.wl_list)

    # 提取各部分参数
    geom_params = data[:, :3]
    state_len = len(config.state_vector_list)
    other_params = data[:, 3 + state_len:]

    basic_params = state_vector[:5]
    remaining_params = state_vector[-3:]
    brdf_values = np.array([state_vector[5 + j] for j in range(m)])

    # 构造输出矩阵
    geom_rep = np.repeat(geom_params, m, axis=0)
    other_rep = np.repeat(other_params, m, axis=0)
    basic_rep = np.tile(basic_params, (n * m, 1))
    remaining_rep = np.tile(remaining_params, (n * m, 1))
    brdf_rep = np.tile(brdf_values, n).reshape(-1, 1)

    data_updated = np.concatenate([geom_rep, basic_rep, brdf_rep, remaining_rep, other_rep], axis=1)
    return data_updated


def predict_multi_wavelength(model_dict: Dict, data_scaled: np.ndarray,
                             config: RetrievalConfig, features_scaler) -> Tuple[np.ndarray, np.ndarray]:
    """
    多波长预测 (PyTorch version)
    此函数被修改为使用PyTorch模型进行推理和雅可比计算
    """
    n_angles = data_scaled.shape[0] // len(config.wl_list)
    K = len(config.state_vector_list)

    # 计算输出维度
    total_cols = sum(2 if config.has_polarization[wl] else 1 for wl in config.wl_list)

    r_sim_all = np.zeros((n_angles, total_cols))
    Kp_all = np.zeros((n_angles, total_cols, K))

    current_col = 0
    for i, wl in enumerate(config.wl_list):
        model, target_scaler = model_dict[wl]

        # 提取当前波长数据
        wl_indices = list(range(i, data_scaled.shape[0], len(config.wl_list)))
        data_for_model = data_scaled[wl_indices].astype(np.float32)

        # --- 1. 使用PyTorch进行预测 ---
        # a. 将numpy数据转为PyTorch张量
        input_tensor = torch.from_numpy(data_for_model).float()

        # b. 使用 no_grad() 上下文进行推理，提高效率
        with torch.no_grad():
            y_pred_scaled_tensor = model(input_tensor)

        # c. 将结果转回numpy数组
        y_pred_scaled = y_pred_scaled_tensor.numpy()
        y_pred = target_scaler.inverse_transform(y_pred_scaled).astype(np.float64)

        # --- 2. 使用PyTorch计算雅可比矩阵 ---
        # a. 定义一个包装函数以供jacobian使用
        def model_func(x):
            return model(x)

        # b. 使用 torch.autograd.functional.jacobian 计算雅可比
        #    它的输出维度为 [batch, output_dim, batch, input_dim]
        jacobian_tensor = torch.autograd.functional.jacobian(model_func, input_tensor, vectorize=True)

        # c. 提取批次维度上的对角线元素，以匹配原始代码的逻辑
        #    结果维度变为 [batch, output_dim, input_dim]
        jacobian_full_tensor = torch.diagonal(jacobian_tensor, offset=0, dim1=0, dim2=2).permute(2, 0, 1)
        jacobian_full = jacobian_full_tensor.detach().numpy()

        # --- 3. 后续处理 (与原代码相同) ---
        #    此部分代码完全基于numpy，因此无需修改
        if config.has_polarization[wl]:
            brdf_start_idx = 8
            r_sim_all[:, current_col:current_col + 2] = y_pred

            # 雅可比矩阵处理
            s_y = target_scaler.scale_.astype(np.float64)
            jacobian_output_adjusted = jacobian_full * s_y.reshape(1, -1, 1)
            s_x = features_scaler.scale_.astype(np.float64)
            jacobian_actual = jacobian_output_adjusted / s_x.reshape(1, 1, -1)

            # 构建完整雅可比
            full_jacobian = np.zeros((n_angles, 2, K))
            full_jacobian[:, :, :5] = jacobian_actual[:, :, 3:brdf_start_idx]
            full_jacobian[:, :, 5 + i] = jacobian_actual[:, :, brdf_start_idx]
            full_jacobian[:, :, 5 + len(config.wl_list):] = jacobian_actual[:, :, brdf_start_idx + 1:brdf_start_idx + 4]

            Kp_all[:, current_col:current_col + 2, :] = full_jacobian
            current_col += 2
        else:
            brdf_start_idx = 8
            r_sim_all[:, current_col] = y_pred[:, 0]

            # 雅可比矩阵处理
            s_y = target_scaler.scale_[0].astype(np.float64)
            jacobian_output_adjusted = jacobian_full[:, 0, :] * s_y
            s_x = features_scaler.scale_.astype(np.float64)
            jacobian_actual = jacobian_output_adjusted / s_x

            # 构建完整雅可比
            full_jacobian = np.zeros((n_angles, K))
            full_jacobian[:, :5] = jacobian_actual[:, 3:brdf_start_idx]
            full_jacobian[:, 5 + i] = jacobian_actual[:, brdf_start_idx]
            full_jacobian[:, 5 + len(config.wl_list):] = jacobian_actual[:, brdf_start_idx + 1:brdf_start_idx + 4]

            Kp_all[:, current_col, :] = full_jacobian
            current_col += 1

    return r_sim_all, Kp_all


# =============== PYTORCH MODIFICATION END ===============


def current_total_cost_function_multi(state_vector: np.ndarray, data: np.ndarray,
                                      r_obs: np.ndarray, model_dict: Dict,
                                      features_scaler, config: RetrievalConfig) -> Tuple[float, np.ndarray, np.ndarray]:
    """计算成本函数"""
    data_updated = update_state_in_data(data, state_vector, config)
    data_scaled = features_scaler.transform(data_updated)
    r_sim, Kp = predict_multi_wavelength(model_dict, data_scaled, config, features_scaler)

    diff = r_obs - r_sim
    n_total = r_obs.size
    weights = compute_weights_improved(r_obs, config)

    # 加权成本
    weighted_diff = weights * (diff ** 2)
    cost = np.sum(weighted_diff) / n_total

    return cost, r_sim, Kp


def print_retrieval_settings(config: RetrievalConfig):
    """打印当前反演设置摘要"""
    print("\n" + "=" * 60)
    print("POLDER-3 气溶胶反演系统配置")
    print("=" * 60)

    print("\n优化设置:")
    print(f"  - 方法: {config.optimization_method}")
    print(f"  - 最大迭代: {config.max_iterations}")
    print(f"  - 参数容差: {config.xtol}")

    print("\n先验设置:")
    print(f"  - 使用先验: {config.use_prior}")
    print(f"  - 先验权重: {config.prior_weight}")
    print(f"  - 正则化权重: {config.regularization_weight}")

    print("\n先验类型:")
    for param, ptype in config.prior_type.items():
        if ptype == "guess":
            print(f"  - {param}: 初始猜测 (几乎无约束)")
        elif ptype == "observation":
            print(f"  - {param}: 观测数据 (强约束)")
        elif ptype == "climatology":
            print(f"  - {param}: 气候学 (中等约束)")

    print("\n" + "=" * 60 + "\n")




def process_pixel_batch_computation(batch_data, config, model_dict_path: str,
                                    features_scaler_path: str, prior_path: str,
                                    polder_time: str) -> List[Dict]:
    """
    纯计算函数，用于多进程处理 (已修复线程冲突问题)
    输入已经预处理好的数据，进行反演计算
    """
    # ======================== 关键性能修复：禁用底层多线程 ========================
    # 在任何计算代码执行之前，立即设置环境变量。
    # 这会强制NumPy/MKL/PyTorch等库在当前这个子进程中只使用单个线程。
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    # ==========================================================================

    try:
        # 在子进程中加载模型
        model_dict = declare_multi_model(config.wl_list, model_dict_path, device='cpu')
        features_scaler = joblib.load(features_scaler_path)

        results = []

        # 获取批次内的原始经纬度信息
        batch_lons = batch_data['lon_list']
        batch_lats = batch_data['lat_list']

        for i in range(len(batch_data['data_obs_list'])):
            try:
                data_obs = batch_data['data_obs_list'][i]
                elev = batch_data['elev_list'][i]
                lon = batch_lons[i]
                lat = batch_lats[i]

                # 执行反演计算
                result = process_single_pixel_improved(
                    i, data_obs, elev, lon, lat, prior_path,
                    config, model_dict, features_scaler, priori_flag=True
                )

                # 为成功的结果添加额外信息
                if 'optimized_state' in result:
                    result['time'] = polder_time
                    result['lon'] = lon  # 确保结果中包含经纬度
                    result['lat'] = lat

                results.append(result)

            except Exception as e:
                # 为失败的结果也附上经纬度，便于调试
                results.append({
                    'pixel_index': i,
                    'error': str(e),
                    'lon': batch_lons[i],
                    'lat': batch_lats[i]
                })

        # 清理内存
        del model_dict, features_scaler
        gc.collect()
        pid = os.getpid()
        num_pixels_in_batch = len(batch_data['data_obs_list'])

        # 使用 f-string 格式化输出，并用 flush=True 强制立即显示
        print(f"[进程 {pid}] ✅ 完成一个批次的处理，包含 {num_pixels_in_batch} 个像元", flush=True)
        return results

    except Exception as e:
        print(f"批次计算失败: {e}")
        # 返回一个包含错误信息的列表，以便主进程知道这个批次失败了
        return [{'pixel_index': -1, 'error': f'Batch-level failure: {e}'}]


def optimize_process_count(total_pixels: int, available_memory_gb: float) -> int:
    """根据像元数和可用内存优化进程数"""
    cpu_count = mp.cpu_count()

    # 基于内存的限制（假设每个进程需要约1-2GB内存）
    memory_based_max = max(1, int(available_memory_gb * 0.85))  # 保守估计，使用40%内存

    # 基于像元数的限制（避免过度分割）
    pixel_based_max = max(1, min(cpu_count, total_pixels // 500))  # 每500像元一个进程

    # 取两者的最小值，但不超过CPU核心数-1
    optimal_processes = min(cpu_count - 1, memory_based_max, pixel_based_max)

    return max(1, optimal_processes)


def create_data_batches(all_pixel_data, n_processes: int) :
    """将数据分割成批次"""
    n_valid_pixels = len(all_pixel_data['data_obs_list'])
    batch_size = max(1, n_valid_pixels // n_processes)

    batches = []
    for i in range(0, n_valid_pixels, batch_size):
        end_idx = min(i + batch_size, n_valid_pixels)
        batch = {
            'data_obs_list': all_pixel_data['data_obs_list'][i:end_idx],
            'elev_list': all_pixel_data['elev_list'][i:end_idx],
            'lon_list': all_pixel_data['lon_list'][i:end_idx],
            'lat_list': all_pixel_data['lat_list'][i:end_idx],
        }
        batches.append(batch)

    return batches


# def create_coordinate_mapping(lons: np.ndarray, lats: np.ndarray) -> Dict:
#     """
#     创建坐标映射关系
#     这个版本有问题 在分grid反演时会将不同位置的像元分到同一个grid里
#     """
#     coordinates = []
#     coordinate_to_lonlat = {}
#
#     for i in range(lons.shape[0]):
#         for j in range(lons.shape[1]):
#             lin, col = calculate_row_col(lons[i, j], lats[i, j])
#             coordinates.append((lin, col))
#             coordinate_to_lonlat[(lin, col)] = (lons[i, j], lats[i, j])
#
#     return coordinates, coordinate_to_lonlat


def update_lonlat_info() -> None:
    """更新经纬度信息"""
    # 这里需要根据实际情况建立像元索引到坐标的映射
    # 由于矢量化提取可能改变了顺序，需要特别处理
    # 简化处理：如果HDF5文件中已有经纬度，则在矢量化提取时已经获取
    pass


def save_results_to_csv(all_results: List[Dict], output_path: str, output_name: str,
                        config: RetrievalConfig) -> str:
    """保存结果到CSV文件"""
    processed_results = []

    for result in all_results:
        if 'error' in result:
            result_data = {
                'time': result.get('time', ''),
                'lon': result.get('lon', np.nan),
                'lat': result.get('lat', np.nan),
                'final_cost': np.nan,
                'vc_total': np.nan,
                'quality_flag': 3,
                'converged': False,
                'error': result['error'],
                **{f"{config.state_vector_list[i]}": np.nan for i in range(config.K)}
            }
        else:
            quality_info = result.get('quality_info', {})
            result_data = {
                'time': result.get('time', ''),
                'lon': result.get('lon', np.nan),
                'lat': result.get('lat', np.nan),
                'final_cost': result.get('final_cost', np.nan),
                'vc_total': quality_info.get('vc_total', np.nan),
                'quality_flag': quality_info.get('quality_flag', 3),
                'converged': quality_info.get('converged', False),
                'error': '',
                **{f"{config.state_vector_list[i]}": result.get('optimized_state', [np.nan] * config.K)[i]
                   for i in range(config.K) if 'optimized_state' in result}
            }
        processed_results.append(result_data)

    # 保存结果
    result_df = pd.DataFrame(processed_results)
    result_file = os.path.join(output_path, output_name)
    result_df.to_csv(result_file, index=False)

    return result_file


def print_performance_summary(io_time: float, compute_time: float, total_time: float,
                              n_pixels: int, n_successful: int):
    """打印性能摘要"""
    print(f"\n{'=' * 60}")
    print("处理性能摘要")
    print(f"{'=' * 60}")
    print(f"总处理时间: {total_time:.2f} 分钟")
    print(f"I/O读取时间: {io_time:.2f} 秒 ({(io_time / 60 / total_time * 100):.1f}%)")
    print(f"计算处理时间: {compute_time:.2f} 秒 ({(compute_time / 60 / total_time * 100):.1f}%)")
    print(f"处理像元数: {n_pixels}")
    print(f"成功像元数: {n_successful}")
    print(f"成功率: {(n_successful / n_pixels * 100):.1f}%")
    print(f"平均处理速度: {n_pixels / total_time:.1f} 像元/分钟")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    """
    优化的POLDER大范围区域反演主函数
    解决I/O瓶颈，实现真正的多进程加速
    """
    T_begin = time()

    # 系统资源检查
    cpu_count = mp.cpu_count()
    available_memory = psutil.virtual_memory().available / (1024 ** 3)
    print(f"系统资源: {cpu_count} CPU核心, {available_memory:.1f}GB 可用内存")

    # 创建配置对象
    config = RetrievalConfig()
    print(f"反演配置: {config.K}个参数, 最大迭代{config.max_iterations}次")

    # 路径配置
    #file_path = r"/media/amers/SSD_part1/whx/ResNet_code/retrieval/test_data/"
    file_path = '/media/amers/Seagate Backup Plus Drive/2006/2006_05_08/'
    file_name_list = [
        'POLDER3_L1B-BG1-030060M_2006-03-18T02-14-20_V1-01.h5',
        'POLDER3_L1B-BG1-030061M_2006-03-18T03-53-13_V1-01.h5',
        'POLDER3_L1B-BG1-030062M_2006-03-18T05-32-07_V1-01.h5',
        'POLDER3_L1B-BG1-030063M_2006-03-18T07-11-01_V1-01.h5',
        'POLDER3_L1B-BG1-030075M_2006-03-19T02-57-40_V1-01.h5',
        'POLDER3_L1B-BG1-030076M_2006-03-19T04-36-34_V1-01.h5',
        'POLDER3_L1B-BG1-030077M_2006-03-19T06-15-28_V1-01.h5',
        'POLDER3_L1B-BG1-030078M_2006-03-19T07-54-23_V1-01.h5'
    ]
    output_path = r'/media/amers/WHX/NNOE_POLDER/retrieval/region/test_results/'
    os.makedirs(output_path, exist_ok=True)

    # 模型路径
    base_dir = '/media/amers/SSD_part1/whx/ResNet_code/forward/resnet_param/'
    features_scaler_path = os.path.join(base_dir, 'scaler_features.pkl')
    prior_path = '/media/amers/WHX/NNOE_POLDER/POLDER_data/priori_data/GRASP_priori_data_05.nc'

    # 处理每个文件
    for file_name in os.listdir(file_path):  #file_name_list
        print(f'\n开始处理文件：{file_name}')

        # 提取时间信息
        polder_time = file_name.split('_')[2].split('T')[0]
        output_name = f"{file_name.split('_')[2]}_single_indian.csv"

        # 定义研究区域（京津冀或其他大范围区域）
        lat_max, lat_min = 39, 5
        lon_min, lon_max = -18, 95

        print(f"处理区域: 经度{lon_min}°-{lon_max}°, 纬度{lat_min}°-{lat_max}°")

        # # 生成完整的坐标范围
        # lat_range = np.arange(lat_max * 100, lat_min * 100, -5) * 0.01
        # lon_range = np.arange(lon_min * 100, lon_max * 100, 5) * 0.01
        # lons, lats = np.meshgrid(lon_range, lat_range)
        #
        # total_grid_points = len(lat_range) * len(lon_range)
        # print(f"网格点总数: {total_grid_points}")
        #
        # # 计算所有像元的行列号
        # print("计算像元坐标...")
        # coordinates, coordinate_to_lonlat = create_coordinate_mapping(lons, lats)
        # print(f"坐标计算完成: {len(coordinates)} 个坐标")
        #
        # # **关键优化1：一次性矢量化读取所有数据**
        # print("开始矢量化I/O读取...")
        # io_start = time()
        print("开始从源文件读取有效像元坐标...")
        io_start = time()

        try:
            with h5py.File(os.path.join(file_path,file_name), 'r') as f:
                # 读取经纬度数据
                lons_all = f['Geolocation_Fields']['Longitude'][:]
                lats_all = f['Geolocation_Fields']['Latitude'][:]

                # 创建区域掩码
                mask = (lons_all >= lon_min) & (lons_all <= lon_max) & \
                       (lats_all >= lat_min) & (lats_all <= lat_max)

                # 获取掩码内像元的行列号 (indices)
                valid_indices = np.argwhere(mask)
                if valid_indices.shape[0] == 0:
                    print("指定区域内没有找到有效的POLDER像元。")
                    continue

                # 将行列号转换为（行，列）元组列表
                coordinates = [tuple(idx) for idx in valid_indices]

        except Exception as e:
            print(f"读取HDF5文件失败: {e}")
            continue

        print(f"坐标读取完成，找到 {len(coordinates)} 个有效像元。")

        all_pixel_data = vectorized_extract_polder_data(
            file_path, file_name, coordinates
        )

        io_time = time() - io_start
        valid_pixels = len(all_pixel_data['data_obs_list'])
        print(f"I/O读取完成，耗时: {io_time:.2f}秒")
        print(f"有效像元数: {valid_pixels} / {len(coordinates)} ({valid_pixels / len(coordinates) * 100:.1f}%)")

        if valid_pixels == 0:
            print("没有有效像元，跳过处理")
            continue

        # 更新经纬度信息（如果需要）
        # update_lonlat_info(all_pixel_data, coordinate_to_lonlat)


        optimal_processes = 94    #optimize_process_count(valid_pixels, available_memory)
        print(f"使用 {optimal_processes} 个进程进行计算")

        # **关键优化3：数据分割，准备多进程计算**
        print("准备数据分割...")
        batches = create_data_batches(all_pixel_data, optimal_processes)
        print(f"数据分割为 {len(batches)} 个批次")

        # 显示批次信息
        for i, batch in enumerate(batches):
            batch_size = len(batch['data_obs_list'])
            print(f"  批次 {i + 1}: {batch_size} 个像元")

        # **关键优化4：多进程纯计算**
        print("开始多进程计算...")
        compute_start = time()
        all_results = []

        with ProcessPoolExecutor(max_workers=optimal_processes) as executor:
            # 提交所有任务
            future_to_batch = {
                executor.submit(process_pixel_batch_computation, batch, config,
                                base_dir, features_scaler_path, prior_path, polder_time): i
                for i, batch in enumerate(batches)
            }

            # 收集结果
            completed_batches = 0
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    completed_batches += 1

                    # 进度报告
                    progress = (completed_batches / len(batches)) * 100
                    processed_pixels = sum(len(b['data_obs_list']) for b in batches[:completed_batches])
                    print(f"完成批次 {batch_idx + 1}, 进度: {progress:.1f}% "
                          f"({processed_pixels}/{valid_pixels} 像元)")

                except Exception as e:
                    print(f"批次 {batch_idx + 1} 处理失败: {e}")

        compute_time = time() - compute_start
        print(f"计算完成，耗时: {compute_time:.2f}秒")

        # **结果保存和后处理**
        if all_results:
            print("保存结果...")
            result_file = save_results_to_csv(all_results, output_path, output_name, config)

            # 计算成功率
            successful_pixels = sum(1 for r in all_results if r.get('error', '') == '')

            print(f"结果保存至: {result_file}")
            print(f"处理统计: 成功 {successful_pixels}/{len(all_results)} 像元")

            # AOD等派生量计算
            try:
                predict_AOD(result_file, base_dir)
            except Exception as e:
                print(f"AOD预测失败: {e}")

        # 清理内存
        del all_pixel_data, all_results, batches
        gc.collect()

        # **性能总结**
        # total_time = (time() - T_begin) / 60
        #
        # print_performance_summary(
        #     io_time, compute_time, total_time,
        #     valid_pixels, successful_pixels if 'successful_pixels' in locals() else 0
        # )

        print("\n处理完成！矢量化I/O + 多进程计算架构运行成功。")