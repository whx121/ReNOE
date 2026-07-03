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
import argparse
import sys
from datetime import datetime

from scipy import sparse
import sys
# 设置环境
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 导入自定义模块
from retrieval.moudle.ResNet_RTModel_pytorch import DNNModel
import retrieval.moudle.moudle_OE as OEfunc
from retrieval.moudle.predict_forAOD import predict_AOD_AAOD_SSA as predict_AOD
from aeronet_inversion_single_pixel_pytorch import compute_weights_improved,current_total_cost_function_multi,declare_multi_model,parallel_pixel_inversion_improved,quality_check

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
    sigma_NN_I: List[float] = field(default_factory=lambda: [0.0006, 0.0007, 0.0006, 0.0007, 0.0007, 0.0005])
    sigma_NN_DOLP: List[float] = field(default_factory=lambda: [0.0, 0.0008, 0.0, 0.0008, 0.0008, 0.0])

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

    # ========== 多像元约束设置 ==========
    use_multi_pixel: bool = True  # 是否使用多像元约束
    pixel_window_size: int = 2  # 邻域窗口大小 (3x3 或 2x2)

    # 空间平滑权重（针对不同参数的二阶导数约束权重）
    smoothness_weights: Dict[str, float] = field(default_factory=lambda: {
        # 气溶胶成分：强平滑约束（假设在小区域内相对均匀）
        "vc_BB": 1,
        "vc_Urban": 1,
        "vc_Ocean": 1,
        "vc_Dust": 1,

        # 气溶胶层高度：中等平滑约束
        "ALH": 10,

        # 地表反照率：弱平滑约束（地表可能有较大变化）
        "iso_0.443": 0,
        "iso_0.49": 0,
        "iso_0.565": 0,
        "iso_0.67": 0,
        "iso_0.865": 0,
        "iso_1.02": 0,

        # BRDF参数：中等平滑约束
        "k1": 0,
        "k2": 0,
        "BPDF": 0
    })

    # 总体平滑约束权重
    smoothness_weight_global: float = 1  # 可调节总体平滑强度

    # 先验类型：区分真实先验和初始猜测
    prior_type: Dict[str, str] = field(default_factory=lambda: {
        "vc_BB": "guess",
        "vc_Urban": "guess",
        "vc_Ocean": "guess",
        "vc_Dust": "guess",
        "ALH": "model",
        "iso": "climatology",
        "k1": "climatology",
        "k2": "climatology",
        "BPDF": "model"
    })

    # 先验不确定性
    prior_sigma: Dict[str, float] = field(default_factory=lambda: {
        "vc_BB": 0.5,
        "vc_Urban": 0.5,
        "vc_Ocean": 0.5,
        "vc_Dust": 0.5,
        "ALH": 0.5,
        "iso": 0.1,
        "k1": 0.1,
        "k2": 0.1,
        "BPDF": 0.3
    })

    prior_weight_factor: Dict[str, float] = field(default_factory=lambda: {
        "observation": 1.0,
        "climatology": 1.0,
        "guess": 0.000000000001,
        "model": 1.0
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
            (0.00001, 0.8),  # vc_BB
            (0, 0.8),  # vc_Urban
            (0, 0.8),  # vc_Ocean
            (0.00001, 0.7),  # vc_Dust
            (0.5, 7),  # ALH
            (0.001, 0.8),  # iso_0.443
            (0.001, 0.8),  # iso_0.490
            (0.001, 0.8),  # iso_0.565
            (0.001, 0.8),  # iso_0.67
            (0.005, 0.8),  # iso_0.865
            (0.005, 0.8),  # iso_1.02
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

        if 'init_state' in config_dict:
            config_dict['init_state'] = np.array(config_dict['init_state'])

        return cls(**config_dict)


@dataclass
class MultiPixelData:
    """多像元数据容器"""
    pixel_indices: List[Tuple[int, int]]
    lon_list: List[float]
    lat_list: List[float]
    elev_list: List[float]
    r_obs_list: List[np.ndarray]
    non_state_list: List[np.ndarray]
    updated_configs: List['RetrievalConfig']  # 新增：每个像元的更新配置
    n_pixels: int
    window_shape: Tuple[int, int]

    def get_pixel_neighbors(self, idx: int) -> List[int]:
        """获取指定像元的邻居索引"""
        i, j = self.pixel_indices[idx]
        neighbors = []

        # 定义邻居偏移（上下左右）
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for di, dj in offsets:
            ni, nj = i + di, j + dj
            # 检查是否在窗口内
            if 0 <= ni < self.window_shape[0] and 0 <= nj < self.window_shape[1]:
                # 查找对应的像元索引
                for k, (pi, pj) in enumerate(self.pixel_indices):
                    if pi == ni and pj == nj:
                        neighbors.append(k)
                        break

        return neighbors


# 确保在文件顶部导入
from scipy import sparse
import sys


class MultiPixelOptimizer:
    """
    多像元联合优化器 (最终版 + 物理尺度归一化)。
    核心：在经过验证的单像素代价函数结构基础上，添加了在“无量纲不确定性空间”
    中计算的空间平滑约束，以消除不同参数物理尺度差异带来的影响。
    """

    def __init__(self, config: RetrievalConfig):
        self.config = config

    def _get_prior_weights_for_pixel(self, pixel_config: RetrievalConfig) -> np.ndarray:
        """获取特定像元配置的先验权重"""
        weights = np.zeros(pixel_config.K)
        for i, param in enumerate(pixel_config.state_vector_list):
            if param.startswith('vc_'):
                base_weight = 1.0 / pixel_config.prior_sigma.get('vc_BB', 1.0)
            elif param.startswith('iso_'):
                base_weight = 1.0 / pixel_config.prior_sigma.get('iso', 0.1)
            elif param in pixel_config.prior_sigma:
                base_weight = 1.0 / pixel_config.prior_sigma[param]
            else:
                base_weight = 1.0
            param_base = param.split('_')[0] if '_' in param else param
            prior_type = pixel_config.prior_type.get(param_base, 'guess')
            type_factor = pixel_config.prior_weight_factor.get(prior_type, 0.01)
            weights[i] = base_weight * type_factor
        return weights

    def _get_sigma_a_for_pixel(self, pixel_config: RetrievalConfig) -> np.ndarray:
        """
        为一个像素的配置，构建其先验不确定性(sigma_a)向量。
        这是实现尺度归一化的关键。
        """
        sigma_a_list = []
        for param in pixel_config.state_vector_list:
            if param.startswith('iso_'):
                sigma_a_list.append(pixel_config.prior_sigma.get('iso', 0.1))
            else:
                param_base = param.split('_')[0] if '_' in param else param
                sigma_a_list.append(pixel_config.prior_sigma.get(param_base, 1.0))

        sigma_a_vector = np.array(sigma_a_list)
        sigma_a_vector[sigma_a_vector == 0] = 1.0  # 防止除以零
        return sigma_a_vector

    def compute_spatial_second_derivative_normalized(self, states: np.ndarray,
                                                     multi_pixel_data: MultiPixelData) -> np.ndarray:
        """
        计算空间二阶导数（拉普拉斯算子），在归一化的状态空间中进行。
        """
        n_pixels = multi_pixel_data.n_pixels
        K = self.config.K
        laplacian = np.zeros((n_pixels, K))

        # === 核心修正 1：首先对状态向量进行逐像素的归一化 ===
        # state_vectors 的形状是 (n_pixels, K)
        states_normalized = np.zeros_like(states)
        for idx in range(n_pixels):
            pixel_config = multi_pixel_data.updated_configs[idx]
            sigma_a_pixel = self._get_sigma_a_for_pixel(pixel_config)
            states_normalized[idx] = states[idx] / sigma_a_pixel

        # 在归一化后的空间中计算拉普拉斯算子
        for idx in range(n_pixels):
            neighbors = multi_pixel_data.get_pixel_neighbors(idx)
            n_neighbors = len(neighbors)
            if n_neighbors > 0:
                for neighbor_idx in neighbors:
                    laplacian[idx] += (states_normalized[neighbor_idx] - states_normalized[idx])
                laplacian[idx] /= n_neighbors

        return laplacian

    def _compute_smoothness_jacobian_sparse_normalized(self, states: np.ndarray,
                                                       multi_pixel_data: MultiPixelData) -> sparse.csr_matrix:
        """计算空间平滑约束的雅可比矩阵 (高效稀疏版本)，并应用归一化。"""
        n_pixels = multi_pixel_data.n_pixels
        K = self.config.K
        n_total_states = n_pixels * K

        jac = sparse.lil_matrix((n_pixels * K, n_total_states))

        for idx in range(n_pixels):
            neighbors = multi_pixel_data.get_pixel_neighbors(idx)
            n_neighbors = len(neighbors)
            if n_neighbors > 0:
                # === 核心修正 2：雅可比也需要被 sigma_a 缩放 ===
                pixel_config = multi_pixel_data.updated_configs[idx]
                sigma_a_pixel = self._get_sigma_a_for_pixel(pixel_config)

                for k, param_name in enumerate(self.config.state_vector_list):
                    param_weight = self.config.smoothness_weights.get(param_name, 0.0)
                    weight = self.config.smoothness_weight_global * param_weight

                    sigma_k = sigma_a_pixel[k]

                    row = idx * K + k
                    col_center = idx * K + k
                    # 根据链式法则 d/dx f(x/a) = f'(x/a) * (1/a)
                    jac[row, col_center] = -weight / sigma_k

                    for neighbor_idx in neighbors:
                        col_neighbor = neighbor_idx * K + k
                        jac[row, col_neighbor] = (weight / n_neighbors) / sigma_k

        return jac.tocsr()

    def _build_single_pixel_data(self, state_vector: np.ndarray,
                                 non_state_matrix: np.ndarray) -> np.ndarray:
        """构建单像元数据矩阵"""
        n_rows = non_state_matrix.shape[0]
        K = self.config.K
        data_matrix = np.zeros((n_rows, 3 + K + (non_state_matrix.shape[1] - 3)))
        data_matrix[:, :3] = non_state_matrix[:, :3]
        data_matrix[:, 3:3 + K] = np.tile(state_vector, (n_rows, 1))
        data_matrix[:, 3 + K:] = non_state_matrix[:, 3:]
        return data_matrix

    def _adjust_init_with_prior(self, init_state: np.ndarray,
                                lon: float, lat: float) -> np.ndarray:
        """没有使用"""
        return init_state

    def optimize_multi_pixel(self, multi_pixel_data: MultiPixelData,
                             model_dict: Dict, features_scaler) -> Dict:
        n_pixels = multi_pixel_data.n_pixels
        K = self.config.K

        init_states = np.array([cfg.init_state for cfg in multi_pixel_data.updated_configs])
        init_state_flat = init_states.flatten()

        bounds_lower = np.concatenate([[b[0] for b in cfg.state_bounds] for cfg in multi_pixel_data.updated_configs])
        bounds_upper = np.concatenate([[b[1] for b in cfg.state_bounds] for cfg in multi_pixel_data.updated_configs])

        def residual_function(state_flat):
            states = state_flat.reshape((n_pixels, K))
            residuals_list = []

            # 1. 逐像素构建基础残差 (观测 + 先验 + 正则化)
            for idx in range(n_pixels):
                sv = states[idx]
                r_obs = multi_pixel_data.r_obs_list[idx]
                pixel_config = multi_pixel_data.updated_configs[idx]
                data = self._build_single_pixel_data(sv, multi_pixel_data.non_state_list[idx])

                _, r_sim, _ = current_total_cost_function_multi(sv, data, r_obs, model_dict, features_scaler,
                                                                pixel_config)

                diff = r_obs - r_sim
                n_total = r_obs.size
                sqrt_weights = np.sqrt(compute_weights_improved(r_obs, pixel_config))
                obs_res = (1.0 / np.sqrt(n_total)) * (sqrt_weights * diff).ravel()

                prior_state = pixel_config.init_state
                prior_weights = self._get_prior_weights_for_pixel(pixel_config)
                prior_res = pixel_config.prior_weight * prior_weights * (sv - prior_state)

                reg_res = pixel_config.regularization_weight * sv

                pixel_total_res = np.concatenate([
                    obs_res,
                    prior_res if pixel_config.use_prior else [],
                    reg_res if pixel_config.use_regularization else []
                ])
                residuals_list.append(pixel_total_res)

            base_residuals = np.concatenate(residuals_list)

            # 2. 附加经过归一化的空间平滑残差
            if self.config.use_multi_pixel and n_pixels > 1:
                # === 核心修正：调用归一化版本的拉普拉斯算子 ===
                laplacian_normalized = self.compute_spatial_second_derivative_normalized(states, multi_pixel_data)

                smooth_residuals = []
                for k, param_name in enumerate(self.config.state_vector_list):
                    param_weight = self.config.smoothness_weights.get(param_name, 0.0)
                    weight = self.config.smoothness_weight_global * param_weight
                    param_smooth_residuals = weight * laplacian_normalized[:, k]
                    smooth_residuals.append(param_smooth_residuals)

                smooth_residuals = np.concatenate(smooth_residuals)
                return np.concatenate([base_residuals, smooth_residuals])
            else:
                return base_residuals

        def jacobian_function(state_flat):
            states = state_flat.reshape((n_pixels, K))
            jac_blocks = []

            # 1. 逐像素构建基础雅可比
            for idx in range(n_pixels):
                sv = states[idx]
                r_obs = multi_pixel_data.r_obs_list[idx]
                pixel_config = multi_pixel_data.updated_configs[idx]
                data = self._build_single_pixel_data(sv, multi_pixel_data.non_state_list[idx])
                _, _, jacobian = current_total_cost_function_multi(sv, data, r_obs, model_dict, features_scaler,
                                                                   pixel_config)

                n_total = r_obs.size
                sqrt_weights = np.sqrt(compute_weights_improved(r_obs, pixel_config))
                J_obs = -(sqrt_weights / np.sqrt(n_total)).reshape(-1, 1) * jacobian.reshape(-1, K)

                J_prior = pixel_config.prior_weight * np.diag(self._get_prior_weights_for_pixel(pixel_config))
                J_reg = pixel_config.regularization_weight * np.eye(K)

                pixel_total_jac = np.vstack([
                    J_obs,
                    J_prior if pixel_config.use_prior else np.empty((0, K)),
                    J_reg if pixel_config.use_regularization else np.empty((0, K))
                ])
                jac_blocks.append(pixel_total_jac)

            J_base = sparse.block_diag(jac_blocks, format='csr')

            # 2. 附加经过归一化的空间平滑雅可比
            if self.config.use_multi_pixel and n_pixels > 1:
                # === 核心修正：调用归一化版本的雅可比计算函数 ===
                J_smooth = self._compute_smoothness_jacobian_sparse_normalized(states, multi_pixel_data)
                return sparse.vstack([J_base, J_smooth], format='csr')
            else:
                return J_base

        # 执行优化
        try:
            result = least_squares(
                fun=residual_function, jac=jacobian_function, x0=init_state_flat,
                bounds=(bounds_lower, bounds_upper), method=self.config.optimization_method,
                xtol=self.config.xtol, ftol=self.config.ftol, gtol=self.config.gtol,
                max_nfev=self.config.max_iterations, x_scale='jac', verbose=0
            )

            optimized_states = result.x.reshape((n_pixels, K))

            final_costs = []
            for idx in range(n_pixels):
                pixel_config = multi_pixel_data.updated_configs[idx]
                data = self._build_single_pixel_data(optimized_states[idx], multi_pixel_data.non_state_list[idx])
                cost, _, _ = current_total_cost_function_multi(
                    optimized_states[idx], data, multi_pixel_data.r_obs_list[idx],
                    model_dict, features_scaler, pixel_config
                )
                final_costs.append(cost)

            return {
                'success': result.success, 'optimized_states': optimized_states,
                'final_costs': final_costs, 'message': result.message
            }
        except Exception as e:
            print(f"多像元优化失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False, 'optimized_states': init_states,
                'final_costs': [np.nan] * n_pixels, 'message': str(e)
            }

def process_multi_pixel_window(window_data: MultiPixelData, config: RetrievalConfig,
                               model_dict: Dict, features_scaler, prior_path: str) -> List[Dict]:
    """
    处理一个多像元窗口
    返回每个像元的反演结果
    """
    # 创建多像元优化器
    optimizer = MultiPixelOptimizer(config)

    # 执行多像元联合优化
    result = optimizer.optimize_multi_pixel(window_data, model_dict, features_scaler)

    # 构建返回结果
    pixel_results = []
    for idx in range(window_data.n_pixels):
        quality_info = quality_check(
            result['optimized_states'][idx],
            result['final_costs'][idx],
            config
        )

        pixel_result = {
            'lon': window_data.lon_list[idx],
            'lat': window_data.lat_list[idx],
            'optimized_state': result['optimized_states'][idx],
            'final_cost': result['final_costs'][idx],
            'quality_info': quality_info,
            'multi_pixel_optimized': True
        }
        pixel_results.append(pixel_result)

    return pixel_results


def organize_pixels_into_windows(data_obs_list: List, elev_list: List,
                                 lon_list: List, lat_list: List,
                                 prior_path: str, config: RetrievalConfig,  # 添加参数
                                 priori_flag: bool,  # 添加参数
                                 window_size: int = 3) -> List[MultiPixelData]:
    """
    将像元组织成窗口用于多像元处理
    window_size: 窗口大小（2或3）
    """
    n_pixels = len(data_obs_list)

    # 根据经纬度构建空间网格
    unique_lons = sorted(list(set(lon_list)))
    unique_lats = sorted(list(set(lat_list)))

    # 创建像元索引映射
    pixel_map = {}
    for idx, (lon, lat) in enumerate(zip(lon_list, lat_list)):
        lon_idx = unique_lons.index(lon)
        lat_idx = unique_lats.index(lat)
        pixel_map[(lat_idx, lon_idx)] = idx

    # 划分窗口
    windows = []
    processed = set()

    for lat_idx in range(0, len(unique_lats), window_size):
        for lon_idx in range(0, len(unique_lons), window_size):
            # 收集窗口内的像元
            window_pixels = []
            window_indices = []

            for di in range(window_size):
                for dj in range(window_size):
                    i = lat_idx + di
                    j = lon_idx + dj

                    if i < len(unique_lats) and j < len(unique_lons):
                        if (i, j) in pixel_map:
                            pixel_idx = pixel_map[(i, j)]
                            if pixel_idx not in processed:
                                window_pixels.append(pixel_idx)
                                window_indices.append((di, dj))
                                processed.add(pixel_idx)

            # 创建窗口数据
            if window_pixels:
                r_obs_list = []
                non_state_list = []
                updated_configs = []  # 新增：保存每个像元的更新配置
                lon_window = []
                lat_window = []
                elev_window = []

                for pixel_idx in window_pixels:
                    # 使用与单像元相同的数据处理函数
                    from aeronet_inversion_single_pixel_pytorch import process_multi_angle_data_improved

                    r_obs, non_state, updated_config = process_multi_angle_data_improved(
                        data_obs_list[pixel_idx], elev_list[pixel_idx],
                        lon_list[pixel_idx], lat_list[pixel_idx],
                        prior_path, config, priori_flag  # 使用真实的配置和先验
                    )

                    r_obs_list.append(r_obs)
                    non_state_list.append(non_state)
                    updated_configs.append(updated_config)
                    lon_window.append(lon_list[pixel_idx])
                    lat_window.append(lat_list[pixel_idx])
                    elev_window.append(elev_list[pixel_idx])

                window_data = MultiPixelData(
                    pixel_indices=window_indices,
                    lon_list=lon_window,
                    lat_list=lat_window,
                    elev_list=elev_window,
                    r_obs_list=r_obs_list,
                    non_state_list=non_state_list,
                    updated_configs=updated_configs,
                    n_pixels=len(window_pixels),
                    window_shape=(min(window_size, len(unique_lats) - lat_idx),
                                  min(window_size, len(unique_lons) - lon_idx))
                )

                windows.append(window_data)

    return windows





def parallel_multi_pixel_inversion(config: RetrievalConfig, data_obs_list: List,
                                   elev_list: List, lon_list: List, lat_list: List,
                                   prior_path: str, output_path: str, output_name: str,
                                   model_dict: Dict, features_scaler, polder_time,
                                   n_jobs: int = -1) -> List[Dict]:
    """
    改进的并行反演函数，支持多像元约束
    """
    n_pixels = len(data_obs_list)
    print(f"开始处理 {n_pixels} 个像元（多像元约束模式）...")

    # 组织像元为窗口
    windows = organize_pixels_into_windows(
        data_obs_list, elev_list, lon_list, lat_list,
        prior_path, config, True,
        config.pixel_window_size
    )

    print(f"组织为 {len(windows)} 个窗口，窗口大小: {config.pixel_window_size}x{config.pixel_window_size}")

    # 并行处理每个窗口
    window_results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_multi_pixel_window)(
            window, config, model_dict, features_scaler, prior_path
        )
        for window in tqdm(windows, desc="Processing windows")
    )

    # 整理结果
    all_results = []
    for window_result in window_results:
        for pixel_result in window_result:
            result_data = {
                'time': polder_time,
                'lon': pixel_result['lon'],
                'lat': pixel_result['lat'],
                'final_cost': pixel_result.get('final_cost', np.nan),
                'vc_total': pixel_result['quality_info'].get('vc_total', np.nan),
                'quality_flag': pixel_result['quality_info'].get('quality_flag', 3),
                'converged': pixel_result['quality_info'].get('converged', False),
                'multi_pixel': pixel_result.get('multi_pixel_optimized', False),
                'error': '',
                **{f"{config.state_vector_list[i]}":
                       pixel_result.get('optimized_state', [np.nan] * config.K)[i]
                   for i in range(config.K)}
            }
            all_results.append(result_data)

    # 保存结果
    result_df = pd.DataFrame(all_results)
    result_file = os.path.join(output_path, output_name)

    if os.path.exists(result_file):
        result_df.to_csv(result_file, mode='a', header=False, index=False)
    else:
        result_df.to_csv(result_file, index=False)

    print(f"处理完成，共 {n_pixels} 个像元")
    print(f"结果保存至: {result_file}")

    return all_results


def debug_print(msg):
    """带时间戳的调试输出"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] DEBUG: {msg}")
    sys.stdout.flush()

def parse_args():
    parser = argparse.ArgumentParser(description='POLDER气溶胶反演 - 多像元约束版本')
    parser.add_argument('--site_index', type=int, required=True,
                       help='指定处理第几个站点（从0开始）')
    parser.add_argument('--sites_csv', type=str,
                       default='/media/amers/WHX/NNOE_POLDER/aeronet/download_aeronet/aeronet_loacation_used_2007.csv',
                       help='站点CSV文件路径')
    parser.add_argument('--n_jobs', type=int, default=16,
                       help='每个站点使用的线程数')
    parser.add_argument('--output_suffix', type=str, default='',
                       help='输出文件后缀，用于区分不同进程')
    parser.add_argument('--window_size', type=int, default=2,
                       help='多像元窗口大小 (2 或 3)')
    parser.add_argument('--smoothness_weight', type=float, default=0.1,
                       help='空间平滑约束权重')
    parser.add_argument('--use_multi_pixel', type=bool, default=True,
                       help='是否启用多像元约束')
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    debug_print("多像元反演脚本开始执行")
    debug_print(
        f"参数: site_index={args.site_index}, window_size={args.window_size}, smoothness_weight={args.smoothness_weight}")

    # 显示当前进程信息
    import os

    debug_print(f"进程PID: {os.getpid()}")

    # 设置进程CPU亲和性信息（可选）
    try:
        import psutil

        p = psutil.Process()
        if hasattr(p, 'cpu_affinity'):
            current_affinity = p.cpu_affinity()
            debug_print(f"当前CPU亲和性: {current_affinity}")
    except:
        pass

    T_begin = time()
    SITES_CSV_PATH = args.sites_csv

    debug_print(f"读取站点CSV文件: {SITES_CSV_PATH}")
    try:
        sites_df = pd.read_csv(SITES_CSV_PATH)
        debug_print(f"成功读取 {len(sites_df)} 个站点")
    except FileNotFoundError:
        debug_print(f"找不到站点CSV文件: {SITES_CSV_PATH}")
        sys.exit(1)

    # 检查站点索引是否有效
    if args.site_index >= len(sites_df):
        debug_print(f"站点索引 {args.site_index} 超出范围 (0-{len(sites_df) - 1})")
        sys.exit(1)

    debug_print("创建多像元配置对象")
    # 创建配置对象，使用命令行参数
    config = RetrievalConfig()
    config.use_multi_pixel = args.use_multi_pixel
    config.pixel_window_size = args.window_size
    config.smoothness_weight_global = args.smoothness_weight

    debug_print("配置信息:")
    debug_print(f"  多像元约束: {config.use_multi_pixel}")
    debug_print(f"  窗口大小: {config.pixel_window_size}x{config.pixel_window_size}")
    debug_print(f"  平滑权重: {config.smoothness_weight_global}")

    # POLDER数据路径
    total_file_path = r"/media/amers/Seagate Backup Plus Drive/2006"
    debug_print(f"检查数据路径: {total_file_path}")
    if not os.path.exists(total_file_path):
        debug_print(f"数据路径不存在: {total_file_path}")
        sys.exit(1)

    date_list = os.listdir(total_file_path)
    debug_print(f"找到 {len(date_list)} 个日期目录")

    # 输出路径
    output_path_ = r'/media/amers/WHX/NNOE_POLDER/retrieval/aeronet/multi_pixel_test/multitest/'

    debug_print("开始加载模型")
    try:
        # 加载模型
        base_dir = '/media/amers/SSD_part1/whx/ResNet_code/forward/resnet_param/'
        model_dict = declare_multi_model(config.wl_list, base_dir, device='cpu')  # 强制使用CPU
        debug_print("模型加载成功")
    except Exception as e:
        debug_print(f"模型加载失败: {e}")
        sys.exit(1)

    debug_print("加载特征缩放器")
    features_scaler = joblib.load(os.path.join(base_dir, 'scaler_features.pkl'))

    # 只处理指定的站点
    row = sites_df.iloc[args.site_index]
    site = row['Site_Name']
    lon = float('%.2f' % row['lon'])
    lat = float('%.2f' % row['lat'])

    debug_print(f"开始处理站点 {args.site_index}: {site} (Lon: {lon}, Lat: {lat})")

    # 创建站点专用的输出目录
    site_output_base = os.path.join(output_path_, f'{site}')
    os.makedirs(site_output_base, exist_ok=True)
    debug_print(f"输出目录: {site_output_base}")

    # 处理该站点的所有月份数据
    processed_files = 0
    for file_dict in date_list:
        try:
            debug_print(f"处理日期目录: {file_dict}")
            file_path = os.path.join(total_file_path, file_dict)
            month = file_dict.split('_')[1]

            # 先验路径
            prior_path = f'/media/amers/WHX/NNOE_POLDER/POLDER_data/priori_data/GRASP_priori_data_{month}.nc'
            output_path = os.path.join(site_output_base, f'{month}')
            os.makedirs(output_path, exist_ok=True)

            # 处理文件
            for file_name in os.listdir(file_path):
                try:
                    debug_print(f'开始处理文件：{file_name}')
                    output_name = f"{site}_retrieval.csv"
                    obs_time = file_name.split('_')[2]

                    # 定义研究区域（扩大区域以支持多像元处理）
                    window_margin = 0.1 #* config.pixel_window_size  # 根据窗口大小调整
                    lat_max, lat_min = lat + window_margin, lat - window_margin - 0.001
                    lon_min, lon_max = lon - window_margin, lon + window_margin + 0.001

                    #debug_print(f"区域范围: Lat({lat_min:.3f}, {lat_max:.3f}), Lon({lon_min:.3f}, {lon_max:.3f})")

                    # 生成格网
                    lat_range = np.arange(lat_max * 100, lat_min * 100, -10) * 0.01
                    lon_range = np.arange(lon_min * 100, lon_max * 100, 10) * 0.01
                    lons, lats = np.meshgrid(lon_range, lat_range)

                    debug_print(f"格网大小: {lons.shape[0]}x{lons.shape[1]} = {lons.size} 个格点")

                    # 初始化数据列表
                    data_obs_list, elev_list, lon_list, lat_list = [], [], [], []

                    debug_print("开始加载像元数据...")
                    # 加载数据
                    for i in range(lons.shape[0]):
                        for j in range(lons.shape[1]):
                            lin, col = OEfunc.calculate_row_col(lons[i, j], lats[i, j])
                            data_obs, cloud, elev, land_sea = OEfunc.extract_polder_h5_(
                                file_path, file_name, lin, col
                            )

                            # 检查数据有效性（无云陆地）
                            if cloud == 1 and land_sea == 100:
                                data_obs_list.append(data_obs)
                                elev_list.append(elev)
                                lon_list.append(lons[i, j])
                                lat_list.append(lats[i, j])

                    debug_print(f'数据加载完成，共 {len(data_obs_list)} 个有效像元')

                    # 执行多像元约束反演
                    if data_obs_list:
                        debug_print("开始执行多像元反演...")
                        if config.use_multi_pixel:
                            # 使用多像元约束反演
                            batch_results = parallel_multi_pixel_inversion(
                                config, data_obs_list, elev_list, lon_list, lat_list,
                                prior_path, output_path, output_name, model_dict,
                                features_scaler, obs_time, n_jobs=args.n_jobs
                            )
                            debug_print("多像元反演完成")
                        else:
                            # 使用原始单像元反演（向后兼容）
                            debug_print("使用单像元模式（向后兼容）")
                            batch_results = parallel_pixel_inversion_improved(
                                config, data_obs_list, elev_list, lon_list, lat_list,
                                prior_path, output_path, output_name, model_dict,
                                features_scaler, obs_time, priori_flag=True, n_jobs=args.n_jobs
                            )

                        # 预测AOD
                        debug_print("开始AOD预测...")
                        predict_AOD(os.path.join(output_path, output_name), base_dir)
                        debug_print("AOD预测完成")

                        processed_files += 1
                        debug_print(f"站点 {site} 月份 {month} 处理完成")
                        break  # 处理完一个文件后就退出（避免重复处理）
                    else:
                        debug_print(f"文件 {file_name} 未覆盖站点 {site}，跳过反演")

                except Exception as e:
                    debug_print(f"处理文件 {file_name} 时发生错误: {e}")
                    import traceback

                    debug_print(f"错误详情: {traceback.format_exc()}")
                    continue

            debug_print(f"站点 {site} 月份 {month} 共处理 {processed_files} 个文件")

        except Exception as e:
            debug_print(f"处理目录 {file_dict} 时发生错误: {e}")
            continue

    T_end = time()
    debug_print(f"站点 {site} 处理完成，总耗时: {(T_end - T_begin) / 60:.2f} 分钟")
    debug_print(f"进程PID {os.getpid()} 完成，共处理 {processed_files} 个文件")