import os
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

# 设置环境
warnings.filterwarnings('ignore', category=UserWarning)

device = torch.device('cpu')
# 导入自定义模块

from retrieval.moudle.ResNet_RTModel_pytorch import DNNModel
import retrieval.moudle.moudle_OE as OEfunc
from retrieval.moudle.predict_forAOD import predict_AOD_AAOD_SSA as predict_AOD


@dataclass
class RetrievalConfig:
    """反演配置类，便于管理和修改参数"""

    # 波长设置
    wl_list: List[str] = field(default_factory=lambda: ["0.443", "0.49", "0.565", "0.67", "0.865"])
    has_polarization: Dict[str, bool] = field(default_factory=lambda: {
        "0.443": False, "0.49": True, "0.565": False,
        "0.67": True, "0.865": True
    })

    # 参数定义
    total_parameters: List[str] = field(default_factory=lambda: [
        "sza", "vza", "fis", "vc_BB", "vc_Urban", "vc_Ocean", "vc_Dust", "ALH",
        "iso_0.443", "iso_0.49", "iso_0.565", "iso_0.67", "iso_0.865",
        "k1", "k2", "BPDF", "o3", "h2o", "dem"
    ])

    # 状态向量（可反演参数）
    state_vector_list: List[str] = field(default_factory=lambda: [
        "vc_BB", "vc_Urban", "vc_Ocean", "vc_Dust", "ALH",
        "iso_0.443", "iso_0.49", "iso_0.565", "iso_0.67", "iso_0.865", 
        "k1", "k2", "BPDF"
    ])

    # 非状态向量（固定参数）
    nonstate_vector_list: List[str] = field(default_factory=lambda: [
        "sza", "vza", "fis", "o3", "h2o", "dem"
    ])

    # 误差设置
    sigma_I: float = 0.05  # 相对测量误差
    sigma_dolp: float = 0.02  # 绝对测量误差
    sigma_NN_I: List[float] = field(default_factory=lambda: [0.001, 0.0011, 0.0008, 0.001, 0.0011])
    sigma_NN_DOLP: List[float] = field(default_factory=lambda: [0.0, 0.0013, 0.0, 0.001, 0.0012])

    # 优化设置
    optimization_method: str = 'trf'  # 'trf', 'dogbox', 'lm'
    max_iterations: int = 30  # 最大迭代次数
    xtol: float = 1e-7  # 变量参数容差
    ftol: float = 1e-7  # 函数容差
    gtol: float = 1e-7  # 梯度容差

    # 正则化设置
    use_regularization: bool = True
    regularization_weight: float = 0.003

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
            (0, 2),  # vc_BB
            (0, 2),  # vc_Urban
            (0, 2),  # vc_Ocean
            (0.00001, 2),  # vc_Dust
            (0.5, 7),  # ALH
            (0.001, 0.8),  # iso_0.443
            (0.001, 0.8),  # iso_0.490
            (0.001, 0.8),  # iso_0.565
            (0.001, 0.8),  # iso_0.67
            (0.005, 0.8),  # iso_0.865
            (0.01, 2),  # k1
            (0.01, 2),  # k2
            (0.5, 8)  # BPDF
        ]

    def _init_state(self):
        """初始化状态向量"""
        self.init_state = np.array([
            0.1,  # vc_BB
            0.1,  # vc_Urban
            0.1,  # vc_Ocean
            0.1,  # vc_Dust
            3.5,  # ALH
            0.05,  # iso_0.443
            0.06,  # iso_0.490
            0.1,  # iso_0.565
            0.12,  # iso_0.67
            0.3,  # iso_0.865
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

def parse_dpc_row_to_data_obs(row: pd.Series) -> np.ndarray:
    """
    将 DPC CSV 的一行转化为原来 POLDER 格式的 data_obs 矩阵。
    处理 Q/U -> DOLP 的转换，并剔除无效角度。
    返回的每一行格式: [sza, vza, phi, I443, I490, DOLP490, I565, I670, DOLP670, I865, DOLP865]
    """
    # 定标系数
    cal_coeffs = {
        '490P': 0.945797463,
        '565':  0.998798994,
        '670P': 1.015393182,
        '865P': 0.961277108
    }
    
    data_obs_list = []
    
    for ang_idx in range(12):  # ang0 to ang11
        sza = row.get(f'sza_ang{ang_idx}')
        # 判断观测是否有效（SZA是NaN则认为该角度无观测）
        if pd.isna(sza) or sza <= 0:
            continue
            
        vza = row[f'vza_ang{ang_idx}']
        phi = row[f'phi_ang{ang_idx}']
        
        # 提取反射率（I443无偏振通道，直接提取）
        I443 = row[f'I443_ang{ang_idx}']
        I565 = row[f'I565_ang{ang_idx}']
        
        # 490P 波段定标
        I490 = row.get(f'I490_ang{ang_idx}')
        Q490 = row.get(f'Q490_ang{ang_idx}')
        U490 = row.get(f'U490_ang{ang_idx}')
        if I490 is not None and not pd.isna(I490):
            I490 = I490 * cal_coeffs['490P']
            Q490 = Q490 * cal_coeffs['490P'] if Q490 is not None and not pd.isna(Q490) else 0.0
            U490 = U490 * cal_coeffs['490P'] if U490 is not None and not pd.isna(U490) else 0.0
        
        # 565 波段定标（该波段只有强度I，没有偏振通道）
        if I565 is not None and not pd.isna(I565):
            I565 = I565 * cal_coeffs['565']
        
        # 670P 波段定标
        I670 = row.get(f'I670_ang{ang_idx}')
        Q670 = row.get(f'Q670_ang{ang_idx}')
        U670 = row.get(f'U670_ang{ang_idx}')
        if I670 is not None and not pd.isna(I670):
            I670 = I670 * cal_coeffs['670P']
            Q670 = Q670 * cal_coeffs['670P'] if Q670 is not None and not pd.isna(Q670) else 0.0
            U670 = U670 * cal_coeffs['670P'] if U670 is not None and not pd.isna(U670) else 0.0
        
        # 865P 波段定标
        I865 = row.get(f'I865_ang{ang_idx}')
        Q865 = row.get(f'Q865_ang{ang_idx}')
        U865 = row.get(f'U865_ang{ang_idx}')
        if I865 is not None and not pd.isna(I865):
            I865 = I865 * cal_coeffs['865P']
            Q865 = Q865 * cal_coeffs['865P'] if Q865 is not None and not pd.isna(Q865) else 0.0
            U865 = U865 * cal_coeffs['865P'] if U865 is not None and not pd.isna(U865) else 0.0
        
        # 计算 DOLP = sqrt(Q^2 + U^2) / I
        def calc_dolp(I, Q, U):
            if I is None or pd.isna(I) or I <= 1e-6:
                return 0.0
            Q_val = Q if Q is not None and not pd.isna(Q) else 0.0
            U_val = U if U is not None and not pd.isna(U) else 0.0
            return np.sqrt(Q_val**2 + U_val**2) / I
        
        DOLP490 = calc_dolp(I490, Q490, U490)
        DOLP670 = calc_dolp(I670, Q670, U670)
        DOLP865 = calc_dolp(I865, Q865, U865)
        
        # 组装为单个角度的向量
        angle_obs = [sza, vza, phi, I443, I490, DOLP490, I565, I670, DOLP670, I865, DOLP865]
        data_obs_list.append(angle_obs)
    
    return np.array(data_obs_list)

def process_multi_angle_data_improved(data_obs, elev, lon, lat, prior_path, config: RetrievalConfig,
                                      priori_flag: bool) -> Tuple[np.ndarray, np.ndarray, RetrievalConfig]:
    """改进的多角度数据处理 (针对 DPC 适配版)"""
    n_valid = data_obs.shape[0]

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
        # 注意：假设 OEfunc 里还是按照原版抽取了所有通道(包含1020)
        # 我们在这里接收 1020 但在组装 prior_values 时直接丢弃它即可。
        iso_443, iso_490, iso_565, iso_670, iso_865, iso_1020, k1, k2, BPDF, ALH = \
            OEfunc.extract_prior_data(lon, lat, prior_path)

        # 检查先验数据的合理性 (删掉 iso_1.02)
        prior_values = {
            'iso_0.443': np.clip(iso_443, 0.005, 0.5),
            'iso_0.49': np.clip(iso_490, 0.005, 0.5),
            'iso_0.565': np.clip(iso_565, 0.005, 0.6),
            'iso_0.67': np.clip(iso_670, 0.005, 0.6),
            'iso_0.865': np.clip(iso_865, 0.005, 0.8),
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
            non_state_matrix[:, i] = 340.5  
        elif param == 'h2o':
            non_state_matrix[:, i] = 1 
        elif param == 'k1' and prior_values:
            non_state_matrix[:, i] = prior_values.get('k1', k1)
        elif param == 'k2' and prior_values:
            non_state_matrix[:, i] = prior_values.get('k2', k2)
        elif param == 'BPDF' and prior_values:
            non_state_matrix[:, i] = prior_values.get('BPDF', BPDF)

    if priori_flag and config.use_prior and prior_values:
        new_config = update_config_with_prior(config, prior_values)
        return r_obs_matrix, non_state_matrix, new_config

    return r_obs_matrix, non_state_matrix, config


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
                margin = 0.1
            elif param =='k1':
                margin = 0.2
            elif param == 'k2':
                margin = 0.1
            elif param == 'BPDF':
                margin = 1
            elif param == 'ALH':
                margin = 1
            else:
                margin = 0.5

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
            print(f"警告：参数 {new_config.state_vector_list[i]} 的初始值{new_config.init_state[i]}已被裁剪到边界内")

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


# def parallel_pixel_inversion_improved(config: RetrievalConfig, data_obs_list: List,
#                                       elev_list: List, lon_list: List, lat_list: List, prior_path,
#                                       output_path: str, output_name: str,
#                                       model_dict: Dict, features_scaler, polder_time,
#                                       priori_flag: bool = True, n_jobs: int = -1) -> List[Dict]:
#     """改进的并行反演函数"""
#     # start_time = time()
#     n_pixels = len(data_obs_list)

#     print(f"开始处理 {n_pixels} 个像元...")

#     # 并行处理
#     results = Parallel(n_jobs=n_jobs, backend='loky')(
#         delayed(process_single_pixel_improved)(
#             i, data_obs_list[i], elev_list[i], lon_list[i], lat_list[i], prior_path,
#             config, model_dict, features_scaler, priori_flag
#         )
#         for i in tqdm(range(n_pixels), desc="Processing pixels")
#     )

#     # 处理结果
#     all_results = []
#     for result in results:
#         if 'error' in result:
#             result_data = {
#                 'lon': lon_list[result['pixel_index']],
#                 'lat': lat_list[result['pixel_index']],
#                 'final_cost': np.nan,
#                 'vc_total': np.nan,
#                 'quality_flag': 3,
#                 'error': result['error'],
#                 **{f"{config.state_vector_list[i]}": np.nan for i in range(config.K)}
#             }
#         else:
#             quality_info = result.get('quality_info', {})
#             result_data = {
#                 'time': polder_time,
#                 'lon': lon_list[result['pixel_index']],
#                 'lat': lat_list[result['pixel_index']],
#                 'final_cost': result.get('final_cost', np.nan),
#                 'vc_total': quality_info.get('vc_total', np.nan),
#                 'quality_flag': quality_info.get('quality_flag', 3),
#                 'converged': quality_info.get('converged', False),
#                 'error': '',
#                 **{f"{config.state_vector_list[i]}": result.get('optimized_state', [np.nan] * config.K)[i]
#                    for i in range(config.K)}
#             }
#         all_results.append(result_data)

#     # 保存结果
#     result_df = pd.DataFrame(all_results)
#     result_file = os.path.join(output_path, output_name)

#     # 保存结果
#     if os.path.exists(result_file):
#         result_df.to_csv(result_file, mode='a', header=False, index=False)
#     else:
#         result_df.to_csv(result_file, index=False)

#     print(f"结果保存至: {result_file}")

#     return all_results

### 适应从CSV中读取！###
def parallel_pixel_inversion_improved(config: RetrievalConfig, data_obs_list: List,
                                      elev_list: List, lon_list: List, lat_list: List, 
                                      times_list: List, site_names: List, prior_path_list: List, # 新增 times_list, site_names, prior_path_list
                                      output_path: str, output_name: str,
                                      model_dict: Dict, features_scaler,
                                      priori_flag: bool = True, n_jobs: int = -1) -> List[Dict]:
    """改进的并行反演函数(适配CSV列表)"""
    n_pixels = len(data_obs_list)
    print(f"开始处理 {n_pixels} 个像元...")

    # 并行处理 (注意此处传入了对应像元的 prior_path)
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_single_pixel_improved)(
            i, data_obs_list[i], elev_list[i], lon_list[i], lat_list[i], prior_path_list[i],
            config, model_dict, features_scaler, priori_flag
        )
        for i in tqdm(range(n_pixels), desc="Processing pixels")
    )

    all_results = []
    for result in results:
        idx = result['pixel_index']
        if 'error' in result:
            result_data = {
                'site_name': site_names[idx],
                'time': times_list[idx],
                'lon': lon_list[idx],
                'lat': lat_list[idx],
                'final_cost': np.nan,
                'vc_total': np.nan,
                'quality_flag': 3,
                'error': result['error'],
                **{f"{config.state_vector_list[i]}": np.nan for i in range(config.K)}
            }
        else:
            quality_info = result.get('quality_info', {})
            result_data = {
                'site_name': site_names[idx],
                'time': times_list[idx],
                'lon': lon_list[idx],
                'lat': lat_list[idx],
                'final_cost': result.get('final_cost', np.nan),
                'vc_total': quality_info.get('vc_total', np.nan),
                'quality_flag': quality_info.get('quality_flag', 3),
                'converged': quality_info.get('converged', False),
                'error': '',
                **{f"{config.state_vector_list[i]}": result.get('optimized_state', [np.nan] * config.K)[i]
                   for i in range(config.K)}
            }
        all_results.append(result_data)

    result_df = pd.DataFrame(all_results)
    result_file = os.path.join(output_path, output_name)

    # 这里的保存逻辑改成了单次写入，因为不再按站点套层循环
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


if __name__ == "__main__":
    T_begin = time()
    
    # 替换为你实际的 CSV 路径
    DPC_CSV_PATH = '/media/amers/SSD_part1/whx/ResNet_forDPC/DPC_Aeronet_Finetune2025.csv'
    
    try:
        dpc_df = pd.read_csv(DPC_CSV_PATH)
        print(f"成功读取 DPC CSV 匹配文件，共 {len(dpc_df)} 条记录。")
    except FileNotFoundError:
        print(f"[错误] 找不到 DPC CSV 文件: '{DPC_CSV_PATH}'")
        exit()

    # 创建配置对象
    config = RetrievalConfig()

    # 输出路径设置
    output_path = r'/media/amers/SSD_part1/whx/ResNet_forDPC/'
    os.makedirs(output_path, exist_ok=True)
    output_name = "dpc_retrieval_results.csv"

    # 加载模型 
    # !!重要!! 必须确保这里的 base_dir 下的模型和 scaler 是专门为 DPC (5波段) 训练的
    base_dir = '/media/amers/SSD_part1/whx/ResNet_code/forward/V1/resnet_param/'
    model_dict = declare_multi_model(config.wl_list, base_dir, device=device)
    features_scaler = joblib.load(os.path.join(base_dir, 'scaler_features.pkl'))

    # 初始化批处理列表
    data_obs_list = []
    elev_list = []
    lon_list = []
    lat_list = []
    times_list = []
    site_names = []
    prior_path_list = []

    print("\n===== 开始解析 DPC 观测数据 =====")
    for index, row in tqdm(dpc_df.iterrows(), total=len(dpc_df), desc="Parsing CSV"):
        try:
            # 解析时间，用于获取月份先验文件
            # 如果你的 time 格式是 '2025-07-19 05:52:13.500'
            obs_time = pd.to_datetime(row['sat_time_utc'])
            month_str = f"{obs_time.month:02d}" # 补齐两位数，如 '07'

            # 你的先验数据路径模板，根据实际情况修改
            prior_path = f'/media/amers/WHX/NNOE_POLDER/POLDER_data/priori_data/GRASP_priori_data_{month_str}.nc'
            
            # 使用新增的函数将单行数据转换为矩阵
            data_obs = parse_dpc_row_to_data_obs(row)
            
            # 只有当该像元包含有效观测角度时才加入列表
            if len(data_obs) > 0:
                data_obs_list.append(data_obs)
                # 注意：如果原来 POLDER elev 是公里，CSV如果是米，此处需/1000。假设已同量纲则不需要除。
                elev_list.append(row['elev']) 
                lon_list.append(row['lon'])
                lat_list.append(row['lat'])
                times_list.append(row['sat_time_utc'])
                site_names.append(row['site_name'])
                prior_path_list.append(prior_path)
                
        except Exception as e:
            print(f"解析第 {index} 行数据时发生错误: {e}")
            continue

    print(f"数据解析完成，共提取 {len(data_obs_list)} 个有效像元等待反演。")

    # 执行大批量并行反演
    if data_obs_list:
        batch_results = parallel_pixel_inversion_improved(
            config=config, 
            data_obs_list=data_obs_list, 
            elev_list=elev_list, 
            lon_list=lon_list, 
            lat_list=lat_list, 
            times_list=times_list,
            site_names=site_names,
            prior_path_list=prior_path_list,
            output_path=output_path, 
            output_name=output_name, 
            model_dict=model_dict, 
            features_scaler=features_scaler, 
            priori_flag=True, 
            n_jobs=-1
        )
        
        # 将结果反演成 AOD 
        # 注意: predict_AOD 需要适配 5波段的模型。如果它是为6波段写的，你需要确保 AOD 预测模型也是 5波段版本
        predict_AOD(os.path.join(output_path, output_name), base_dir)
    else:
        print("未提取到任何有效数据。")

    T_end = time()
    print(f"\n程序总运行时间: {(T_end - T_begin) / 60:.2f} 分钟")
    print('完成')