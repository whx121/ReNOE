import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import sys
import json
import warnings
import logging
import argparse
from time import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
import torch
import torch.nn as nn
# ==============================================================================
# 1. 环境与模块配置
# ==============================================================================

# 设置环境
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 导入自定义模块 (请确保这些模块在您的Python路径下)
from retrieval.moudle.ResNet_RTModel_pytorch import DNNModel
import retrieval.moudle.moudle_OE as OEfunc
from retrieval.moudle.predict_forAOD import predict_AOD_AAOD_SSA as predict_AOD

from aeronet_inversion_single_pixel_pytorch import (
    compute_weights_improved,
    current_total_cost_function_multi,
    declare_multi_model,
    process_multi_angle_data_improved,
    quality_check
)


# ==============================================================================
# 2. 数据结构定义 (Dataclasses)
# ==============================================================================

@dataclass
class RetrievalConfig:
    """反演配置类，便于管理和修改参数 (与您的源代码一致)"""

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
    sigma_I: float = 0.03
    sigma_dolp: float = 0.01
    sigma_NN_I: List[float] = field(default_factory=lambda: [0.0008, 0.0009, 0.0009, 0.0009, 0.0009, 0.0007])
    sigma_NN_DOLP: List[float] = field(default_factory=lambda: [0.0, 0.001, 0.0, 0.0009, 0.0008, 0.0])

    # 优化设置
    optimization_method: str = 'trf'
    max_iterations: int = 16
    xtol: float = 1e-7
    ftol: float = 1e-7
    gtol: float = 1e-7

    # 正则化设置
    use_regularization: bool = False
    regularization_weight: float = 0.003

    # 先验设置
    use_prior: bool = False
    prior_weight: float = 1

    # ========== 多像元约束设置 ==========
    use_multi_pixel: bool = True
    pixel_window_size: int = 2  # 仅用于决定查找邻居的数量 (2x2 -> 4个邻居)

    # 空间平滑权重
    smoothness_weights: Dict[str, float] = field(default_factory=lambda: {
        "vc_BB": 1, 
        "vc_Urban": 1,
        "vc_Ocean": 1, 
        "vc_Dust": 1, 
        "ALH": 10,
        "iso_0.443": 0, "iso_0.49": 0, "iso_0.565": 0, "iso_0.67": 0, "iso_0.865": 0, "iso_1.02": 0,
        "k1": 0, "k2": 0, "BPDF": 0
    })
    smoothness_weight_global: float = 1.0

    # 先验类型
    prior_type: Dict[str, str] = field(default_factory=lambda: {
        "vc_BB": "guess",
        "vc_Urban": "guess",
        "vc_Ocean": "guess",
        "vc_Dust": "guess",
        "ALH": "climatology",
        "iso": "climatology", "k1": "climatology", "k2": "climatology", "BPDF": "climatology"
    })

    # 先验不确定性
    prior_sigma: Dict[str, float] = field(default_factory=lambda: {
        "vc_BB": 0.4,
        "vc_Urban": 0.1,
        "vc_Ocean": 0.1,
        "vc_Dust": 0.4,
        "ALH": 0.5, 
        "iso": 0.1, 
        "k1": 0.1, 
        "k2": 0.1, 
        "BPDF": 0.3
    })

    prior_weight_factor: Dict[str, float] = field(default_factory=lambda: {
        "observation": 1.0,
        "climatology": 0.8,
        "guess": 0.2,
        "model": 0.6
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
            (0, 1), 
            (0, 1), 
            (0, 1), 
            (1e-6, 2),
            (0.5, 8),
            (0.0005, 0.8), (0.0005, 0.8), (0.001, 0.8), (0.001, 0.8), (0.005, 0.8), (0.005, 0.8),
            (0.01, 2), (0.01, 2), (0.5, 8)
        ]

    def _init_state(self):
        """初始化状态向量"""
        self.init_state = np.array([
            0.1, 0.00001, 0.00001, 0.00001, 3.5, 0.05, 0.06, 0.1, 0.12, 0.3, 0.4, 0.6, 0.4, 4
        ])

    def _init_obs_count(self):
        """初始化观测数量"""
        self.obs_count_per_wl = {
            wl: 2 if self.has_polarization[wl] else 1
            for wl in self.wl_list
        }
@dataclass
class MultiPixelData:
    """多像元数据容器 (简化版)"""
    lon_list: List[float]
    lat_list: List[float]
    elev_list: List[float]
    r_obs_list: List[np.ndarray]
    non_state_list: List[np.ndarray]
    updated_configs: List['RetrievalConfig']
    n_pixels: int


# ==============================================================================
#  高效I/O与数据提取函数
# ==============================================================================

# def create_coordinate_mapping(lons: np.ndarray, lats: np.ndarray) -> List[Tuple[int, int]]:
#     """创建坐标映射，将经纬度网格转换为POLDER行列坐标"""
#     coordinates = []
#     for i in range(lons.shape[0]):
#         for j in range(lons.shape[1]):
#             lon, lat = lons[i, j], lats[i, j]
#             lin, col = OEfunc.calculate_row_col(lon, lat)
#             coordinates.append((lin, col))
#     return coordinates


def _read_all_dataset_slices(f: h5py.File, r_min, r_max, c_min, c_max) -> Dict[str, np.ndarray]:
    """一次性读取所有HDF5数据集的切片"""
    slices = {}
    r_slice = slice(r_min, r_max + 1)
    c_slice = slice(c_min, c_max + 1)
    datasets_to_read = [
        ('Geolocation_Fields', 'Longitude', 'lons'), ('Geolocation_Fields', 'Latitude', 'lats'),
        ('Geolocation_Fields', 'surface_altitude', 'surface_altitude'),
        ('Geolocation_Fields', 'land_sea_flag', 'land_sea_flag'),
        ('Data_Fields', 'cloud_indicator', 'cloud_indicator'),
        ('Data_Directional_Fields', 'thetas', 'thetas'), ('Data_Directional_Fields', 'thetav', 'thetav'),
        ('Data_Directional_Fields', 'phi', 'phi'), ('Data_Directional_Fields', 'I443NP', 'I443NP'),
        ('Data_Directional_Fields', 'I490P', 'I490P'), ('Data_Directional_Fields', 'Q490P', 'Q490P'),
        ('Data_Directional_Fields', 'U490P', 'U490P'), ('Data_Directional_Fields', 'I565NP', 'I565NP'),
        ('Data_Directional_Fields', 'I670P', 'I670P'), ('Data_Directional_Fields', 'Q670P', 'Q670P'),
        ('Data_Directional_Fields', 'U670P', 'U670P'), ('Data_Directional_Fields', 'I865P', 'I865P'),
        ('Data_Directional_Fields', 'Q865P', 'Q865P'), ('Data_Directional_Fields', 'U865P', 'U865P'),
        ('Data_Directional_Fields', 'I1020NP', 'I1020NP')
    ]
    for group, dset, key in datasets_to_read:
        try:
            slices[key] = f[group][dset][r_slice, c_slice]
        except KeyError:
            shape = (r_max - r_min + 1, c_max - c_min + 1)
            slices[key] = np.full(shape + (() if len(f[group][dset].shape) == 2 else (f[group][dset].shape[2],)),
                                  -32767)
    return slices


def _calculate_observations_from_slice(pixel_data: Dict) -> Optional[np.ndarray]:
    """从内存中的数据切片计算单个像元的观测数据"""
    sza = OEfunc.mean_tif(pixel_data['thetas'], 65534, 0.0015)
    vza = OEfunc.mean_tif(pixel_data['thetav'], 65534, 0.0015)
    phi_raw = OEfunc.mean_tif(pixel_data['phi'], 65534, 0.006)
    phi = np.abs((phi_raw + 180) % 360 - 180)

    cos_sza = np.cos(np.deg2rad(sza))
    cos_sza[cos_sza < 1e-6] = np.nan

    def get_refl(key):
        val = OEfunc.mean_tif(pixel_data[key], -32767, 0.0001)
        return val / cos_sza

    I443, I490, I565, I670, I865, I1020 = map(get_refl, ['I443NP', 'I490P', 'I565NP', 'I670P', 'I865P', 'I1020NP'])
    Q490, U490, Q670, U670, Q865, U865 = map(lambda k: OEfunc.mean_tif(pixel_data[k], -32767, 0.0001),
                                             ['Q490P', 'U490P', 'Q670P', 'U670P', 'Q865P', 'U865P'])

    I490_raw = OEfunc.mean_tif(pixel_data['I490P'], -32767, 0.0001)
    I670_raw = OEfunc.mean_tif(pixel_data['I670P'], -32767, 0.0001)
    I865_raw = OEfunc.mean_tif(pixel_data['I865P'], -32767, 0.0001)

    I490_raw[I490_raw < 1e-6] = np.nan
    I670_raw[I670_raw < 1e-6] = np.nan
    I865_raw[I865_raw < 1e-6] = np.nan

    dolp_490 = np.sqrt(Q490 ** 2 + U490 ** 2) / I490_raw
    dolp_670 = np.sqrt(Q670 ** 2 + U670 ** 2) / I670_raw
    dolp_865 = np.sqrt(Q865 ** 2 + U865 ** 2) / I865_raw

    return OEfunc.package_observation(sza, vza, phi, I443, I490, dolp_490, I565, I670, dolp_670, I865, dolp_865, I1020)


def vectorized_extract_polder_data(file_path: str, file_name: str, coordinates: List[Tuple[int, int]]) -> Dict:
    """【高效版】矢量化提取POLDER数据"""
    full_file_path = os.path.join(file_path, file_name)
    if not coordinates: return {'data_obs_list': [], 'elev_list': [], 'lon_list': [], 'lat_list': []}

    lins = np.array([c[0] for c in coordinates])
    cols = np.array([c[1] for c in coordinates])
    lin_min, lin_max = lins.min(), lins.max()
    col_min, col_max = cols.min(), cols.max()
    target_coords_set = set(coordinates)

    all_data = {'data_obs_list': [], 'elev_list': [], 'lon_list': [], 'lat_list': []}

    with h5py.File(full_file_path, 'r') as f:
        data_slices = _read_all_dataset_slices(f, lin_min, lin_max, col_min, col_max)
        for r_idx in range(data_slices['lons'].shape[0]):
            for c_idx in range(data_slices['lons'].shape[1]):
                abs_lin, abs_col = lin_min + r_idx, col_min + c_idx
                if (abs_lin, abs_col) not in target_coords_set: continue

                pixel_data = {key: val[r_idx, c_idx] for key, val in data_slices.items()}

                cloud_val = OEfunc.cloud_grasp(pixel_data.get('cloud_indicator', 255))
                if cloud_val != 1 or pixel_data.get('land_sea_flag', 0) != 100: continue

                data_obs = _calculate_observations_from_slice(pixel_data)
                if data_obs is not None and data_obs.shape[0] > 0:
                    all_data['data_obs_list'].append(data_obs)
                    all_data['elev_list'].append(pixel_data['surface_altitude'] / 1000.0)
                    all_data['lon_list'].append(pixel_data['lons'])
                    all_data['lat_list'].append(pixel_data['lats'])
    return all_data


# ==============================================================================
# 4. 多像元优化器 (核心)
# ==============================================================================

class MultiPixelOptimizer:
    """多像元联合优化器 (使用k-d树动态查找邻居)"""

    def __init__(self, config: RetrievalConfig):
        self.config = config

    def _get_prior_weights_for_pixel(self, pixel_config: RetrievalConfig) -> np.ndarray:
        weights = np.zeros(pixel_config.K)
        for i, param in enumerate(pixel_config.state_vector_list):
            if param.startswith('vc_'):
                base_param = 'vc_BB'
            elif param.startswith('iso_'):
                base_param = 'iso'
            else:
                base_param = param

            base_weight = 1.0 / pixel_config.prior_sigma.get(base_param, 1.0)
            prior_type_key = param.split('_')[0] if '_' in param else param
            prior_type = pixel_config.prior_type.get(prior_type_key, 'guess')
            type_factor = pixel_config.prior_weight_factor.get(prior_type, 0.01)
            weights[i] = base_weight * type_factor
        return weights

    def _get_sigma_a_for_pixel(self, pixel_config: RetrievalConfig) -> np.ndarray:
        sigma_a_list = []
        for param in pixel_config.state_vector_list:
            if param.startswith('iso_'):
                base_param = 'iso'
            else:
                base_param = param.split('_')[0] if '_' in param else param
            sigma_a_list.append(pixel_config.prior_sigma.get(base_param, 1.0))
        sigma_a_vector = np.array(sigma_a_list)
        sigma_a_vector[sigma_a_vector == 0] = 1.0
        return sigma_a_vector

    def _find_neighbors_kdtree(self, n_pixels: int, lon_list: List[float], lat_list: List[float]) -> List[List[int]]:
        """【新增】使用k-d树为每个像元查找邻居"""
        if n_pixels <= 1: return [[] for _ in range(n_pixels)]
        coords = np.vstack([lon_list, lat_list]).T
        kdtree = cKDTree(coords)

        # 查找邻居数量 = 窗口边长^2 - 1, (e.g., 2x2 -> 3 neighbors, 3x3 -> 8 neighbors)
        k_neighbors = self.config.pixel_window_size ** 2
        k_query = min(k_neighbors, n_pixels)

        _, indices = kdtree.query(coords, k=k_query)

        all_neighbors = []
        for i in range(n_pixels):
            # 排除自身 (总在第一个)
            all_neighbors.append(indices[i, 1:].tolist())
        return all_neighbors

    def _compute_spatial_laplacian(self, states: np.ndarray, multi_pixel_data: MultiPixelData,
                                   neighbors_map: List[List[int]]) -> np.ndarray:
        """计算空间拉普拉斯算子 (忠于原始代码的归一化策略)"""
        n_pixels, K = states.shape
        laplacian = np.zeros_like(states)

        states_normalized = np.zeros_like(states)
        for idx in range(n_pixels):
            pixel_config = multi_pixel_data.updated_configs[idx]
            sigma_a = self._get_sigma_a_for_pixel(pixel_config)
            states_normalized[idx] = states[idx] / sigma_a

        for idx in range(n_pixels):
            neighbors = neighbors_map[idx]
            if neighbors:
                # 核心思想：中心像元的值与邻居平均值的差异
                neighbor_states_mean = np.mean(states_normalized[neighbors, :], axis=0)
                laplacian[idx] = neighbor_states_mean - states_normalized[idx]
        return laplacian

    def _compute_smoothness_jacobian(self, multi_pixel_data: MultiPixelData,
                                     neighbors_map: List[List[int]]) -> sparse.csr_matrix:
        """计算空间平滑约束的雅可比矩阵 (忠于原始代码的归一化策略)"""
        n_pixels = multi_pixel_data.n_pixels
        K = self.config.K
        jac = sparse.lil_matrix((n_pixels * K, n_pixels * K))

        for idx in range(n_pixels):
            neighbors = neighbors_map[idx]
            if neighbors:
                pixel_config = multi_pixel_data.updated_configs[idx]
                sigma_a = self._get_sigma_a_for_pixel(pixel_config)
                for k, param_name in enumerate(self.config.state_vector_list):
                    param_weight = self.config.smoothness_weights.get(param_name, 0.0)
                    weight = self.config.smoothness_weight_global * param_weight

                    row = idx * K + k
                    # 中心像元贡献
                    jac[row, idx * K + k] = -weight / sigma_a[k]
                    # 邻居像元贡献
                    for neighbor_idx in neighbors:
                        jac[row, neighbor_idx * K + k] = (weight / len(neighbors)) / sigma_a[k]
        return jac.tocsr()

    def _build_single_pixel_data(self, state_vector: np.ndarray, non_state_matrix: np.ndarray) -> np.ndarray:
        n_rows = non_state_matrix.shape[0]
        K = self.config.K
        data_matrix = np.zeros((n_rows, 3 + K + (non_state_matrix.shape[1] - 3)))
        data_matrix[:, :3] = non_state_matrix[:, :3]
        data_matrix[:, 3:3 + K] = np.tile(state_vector, (n_rows, 1))
        data_matrix[:, 3 + K:] = non_state_matrix[:, 3:]
        return data_matrix

    def optimize_multi_pixel(self, multi_pixel_data: MultiPixelData, model_dict: Dict, features_scaler) -> Dict:
        """执行多像元联合优化"""
        n_pixels, K = multi_pixel_data.n_pixels, self.config.K

        # 动态构建邻居关系图
        neighbors_map = self._find_neighbors_kdtree(n_pixels, multi_pixel_data.lon_list, multi_pixel_data.lat_list)

        # 准备初始状态和边界
        init_states = np.array([cfg.init_state for cfg in multi_pixel_data.updated_configs])
        init_state_flat = init_states.flatten()
        bounds_lower = np.concatenate([cfg.state_bounds for cfg in multi_pixel_data.updated_configs])[:, 0]
        bounds_upper = np.concatenate([cfg.state_bounds for cfg in multi_pixel_data.updated_configs])[:, 1]

        # 预计算平滑约束的雅可比矩阵（因为它不依赖于状态变量x）
        J_smooth = self._compute_smoothness_jacobian(multi_pixel_data, neighbors_map)

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
                sqrt_weights = np.sqrt(compute_weights_improved(r_obs, pixel_config))
                obs_res = (1.0 / np.sqrt(r_obs.size)) * (sqrt_weights * diff).ravel()

                prior_res = pixel_config.prior_weight * self._get_prior_weights_for_pixel(pixel_config) * (
                            sv - pixel_config.init_state)
                reg_res = pixel_config.regularization_weight * sv

                residuals_list.append(np.concatenate([
                    obs_res,
                    prior_res if pixel_config.use_prior else [],
                    reg_res if pixel_config.use_regularization else []
                ]))

            base_residuals = np.concatenate(residuals_list)

            # 2. 附加空间平滑残差
            if self.config.use_multi_pixel and n_pixels > 1:
                laplacian = self._compute_spatial_laplacian(states, multi_pixel_data, neighbors_map)
                smooth_residuals = []
                for k, param_name in enumerate(self.config.state_vector_list):
                    param_weight = self.config.smoothness_weights.get(param_name, 0.0)
                    weight = self.config.smoothness_weight_global * param_weight
                    smooth_residuals.append(weight * laplacian[:, k])
                return np.concatenate([base_residuals, np.concatenate(smooth_residuals)])
            else:
                return base_residuals

        def jacobian_function(state_flat):
            states = state_flat.reshape((n_pixels, K))
            jac_blocks = []

            # 1. 逐像素构建基础雅可比
            for idx in range(n_pixels):
                sv, r_obs, pixel_config = states[idx], multi_pixel_data.r_obs_list[idx], \
                multi_pixel_data.updated_configs[idx]
                data = self._build_single_pixel_data(sv, multi_pixel_data.non_state_list[idx])
                _, _, jacobian = current_total_cost_function_multi(sv, data, r_obs, model_dict, features_scaler,
                                                                   pixel_config)

                sqrt_weights = np.sqrt(compute_weights_improved(r_obs, pixel_config))
                J_obs = -(sqrt_weights / np.sqrt(r_obs.size)).reshape(-1, 1) * jacobian.reshape(-1, K)
                J_prior = pixel_config.prior_weight * np.diag(self._get_prior_weights_for_pixel(pixel_config))
                J_reg = pixel_config.regularization_weight * np.eye(K)

                jac_blocks.append(np.vstack([
                    J_obs,
                    J_prior if pixel_config.use_prior else np.empty((0, K)),
                    J_reg if pixel_config.use_regularization else np.empty((0, K))
                ]))

            J_base = sparse.block_diag(jac_blocks, format='csr')

            # 2. 附加空间平滑雅可比
            if self.config.use_multi_pixel and n_pixels > 1:
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

            final_costs = [current_total_cost_function_multi(
                optimized_states[i],
                self._build_single_pixel_data(optimized_states[i], multi_pixel_data.non_state_list[i]),
                multi_pixel_data.r_obs_list[i], model_dict, features_scaler, multi_pixel_data.updated_configs[i]
            )[0] for i in range(n_pixels)]

            return {'success': result.success, 'optimized_states': optimized_states, 'final_costs': final_costs}
        except Exception as e:
            print(f"多像元优化失败: {e}")
            return {'success': False, 'optimized_states': init_states, 'final_costs': [np.nan] * n_pixels}





def process_data_block(block_info: Dict) -> List[Dict]:
    """
    处理单个数据批次的完整流程 (由每个CPU核心执行)
    这个函数现在处理的是一个像元列表，而不是一个地理区域
    """
    block_id = block_info['id']
    pixel_batch = block_info['pixel_batch']  # 获取像元数据批次
    prior_path, config, model_dict, features_scaler, polder_time,doy = \
        block_info['prior_path'], block_info['config'], block_info['model_dict'], \
            block_info['features_scaler'], block_info['polder_time'], block_info['doy']

    print(f"[进程 {block_id}] 开始处理一个包含 {len(pixel_batch['lon_list'])} 个像元的批次...")

    try:
        # 1. 预处理批次内的所有像元
        preprocessed_data = {'lon': [], 'lat': [], 'elev': [], 'r_obs': [], 'non_state': [], 'config': []}
        failed_count = 0

        for i in range(len(pixel_batch['lon_list'])):
            try:

                # 注意：这里的 'data_obs_list' 已经在 pixel_batch 中了
                r_obs, non_state, updated_config = process_multi_angle_data_improved(
                    vc_model, scaler_x_vc, scaler_y_vc,
                    pixel_batch['data_obs_list'][i],
                    doy,
                    pixel_batch['elev_list'][i],
                    pixel_batch['lon_list'][i],
                    pixel_batch['lat_list'][i],
                    prior_path,
                    config,
                    True
                )
                if r_obs is None or r_obs.size == 0 or non_state is None or non_state.size == 0:
                    failed_count += 1
                    continue

                preprocessed_data['lon'].append(pixel_batch['lon_list'][i])
                preprocessed_data['lat'].append(pixel_batch['lat_list'][i])
                preprocessed_data['elev'].append(pixel_batch['elev_list'][i])
                preprocessed_data['r_obs'].append(r_obs)
                preprocessed_data['non_state'].append(non_state)
                preprocessed_data['config'].append(updated_config)

            except Exception as e:
                failed_count += 1
                print(f"[进程 {block_id}] 错误: 像元 {i} 预处理失败: {str(e)}") # 可以取消注释以进行详细调试
                continue

        print(f"[进程 {block_id}] 预处理完成: 成功 {len(preprocessed_data['lon'])} 个，失败 {failed_count} 个")

        if not preprocessed_data['lon']:
            return []

        # 2. 将整个块的数据打包成MultiPixelData对象 (这部分逻辑和原来一样)
        block_data = MultiPixelData(
            lon_list=preprocessed_data['lon'],
            lat_list=preprocessed_data['lat'],
            elev_list=preprocessed_data['elev'],
            r_obs_list=preprocessed_data['r_obs'],
            non_state_list=preprocessed_data['non_state'],
            updated_configs=preprocessed_data['config'],
            n_pixels=len(preprocessed_data['lon'])
        )
        print(f"[进程 {block_id}] 开始对 {block_data.n_pixels} 个像元进行联合优化...")

        # 3. 执行多像元联合优化
        optimizer = MultiPixelOptimizer(config)
        result = optimizer.optimize_multi_pixel(block_data, model_dict, features_scaler)

        pixel_results = []
        for idx in range(block_data.n_pixels):
            try:
                quality_info = quality_check(result['optimized_states'][idx], result['final_costs'][idx], config)
                pixel_results.append({
                    'time': polder_time,
                    'lon': block_data.lon_list[idx],
                    'lat': block_data.lat_list[idx],
                    'final_cost': result['final_costs'][idx],
                    'vc_total': quality_info.get('vc_total', np.nan),
                    'quality_flag': quality_info.get('quality_flag', 3),
                    'converged': quality_info.get('converged', False),
                    'multi_pixel': True,
                    'error': '',
                    **{f"{config.state_vector_list[i]}": result['optimized_states'][idx][i] for i in range(config.K)}
                })
            except Exception as e:
                print(f"[进程 {block_id}] 警告: 像元 {idx} 质量检查失败: {str(e)}")
                # 创建一个默认结果，避免丢失数据
                pixel_results.append({
                    'time': polder_time,
                    'lon': block_data.lon_list[idx],
                    'lat': block_data.lat_list[idx],
                    'final_cost': np.nan,
                    'vc_total': np.nan,
                    'quality_flag': 3,
                    'converged': False,
                    'multi_pixel': True,
                    'error': str(e),
                    **{f"{config.state_vector_list[i]}": np.nan for i in range(config.K)}
                })

        print(f"[进程 {block_id}] 处理完成，输出 {len(pixel_results)} 个结果")
        return pixel_results

    except Exception as e:
        print(f"[进程 {block_id}] 致命错误: {str(e)}")
        import traceback
        print(f"[进程 {block_id}] 详细错误: {traceback.format_exc()}")
        return []


def create_pixel_batches(all_pixel_data: Dict, num_batches: int) -> List[Dict]:
    """将从文件中读取的所有像元数据分割成指定数量的批次，用于并行处理"""
    n_total_pixels = len(all_pixel_data['lon_list'])
    if n_total_pixels == 0:
        return []

    # 计算每个批次大概有多少个像元
    batch_size = n_total_pixels // num_batches
    if n_total_pixels % num_batches > 0:
        batch_size += 1

    batches = []
    for i in range(0, n_total_pixels, batch_size):
        end_idx = min(i + batch_size, n_total_pixels)
        batch = {
            'lon_list': all_pixel_data['lon_list'][i:end_idx],
            'lat_list': all_pixel_data['lat_list'][i:end_idx],
            'elev_list': all_pixel_data['elev_list'][i:end_idx],
            'data_obs_list': all_pixel_data['data_obs_list'][i:end_idx]
        }
        batches.append(batch)
    return batches

class ResidualDNN(nn.Module):
    # (您的ResidualDNN类定义，与训练时完全相同)
    def __init__(self, input_dim, output_dim, layers_config=[256, 128, 64],
                 dropout_rate=0.2, l2_regularization=1e-4):
        super(ResidualDNN, self).__init__()
        layers_list = []; current_dim = input_dim
        for i, units in enumerate(layers_config):
            block = nn.Sequential(nn.Linear(current_dim, units), nn.LayerNorm(units), nn.LeakyReLU(0.2), nn.Dropout(dropout_rate))
            layers_list.append(block)
            is_residual = (i != 0 and current_dim == units)
            layers_list.append(is_residual)
            current_dim = units
        self.hidden_layers = nn.ModuleList([layer for layer in layers_list if isinstance(layer, nn.Module)])
        self.residual_flags = [flag for flag in layers_list if isinstance(flag, bool)]
        self.output_layer = nn.Linear(current_dim, output_dim)
    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x_prev_layer = x; x = layer(x)
            if self.residual_flags[i]: x = x + x_prev_layer
        x = self.output_layer(x); return x

def main():
    """主函数 - 大区域多像元并行反演"""
    T_begin = time()

    # ========== 1. 参数配置 ==========
    config = RetrievalConfig()
    config.use_multi_pixel = True
    config.pixel_window_size = 2  # 查找邻居数 = 2*2 = 4 (包含自身)
    config.smoothness_weight_global = 0.05

    # 路径配置
    file_path = r'/media/amers/Seagate Backup Plus Drive/2007/2007_03_29/'

    file_name_list = [
        'POLDER3_L1B-BG1-053179M_2007-03-29T06-25-44_V1-01.h5',


    ]

    output_path = r'/media/amers/WHX/NNOE_POLDER/retrieval/region/test_results_large_area/'
    prior_path = '/media/amers/WHX/NNOE_POLDER/POLDER_data/priori_data/GRASP_priori_data_03.nc'
    base_dir = '/media/amers/SSD_part1/whx/ResNet_code/forward/V1/resnet_param/'



    # 定义研究区域
    lat_max, lat_min = 28, 8
    lon_min, lon_max = 86, 110

    # 并行设置
    NUM_PROCESSES = 94  # 先用较少的进程数测试

    # ========== 2. 初始化 ==========
    os.makedirs(output_path, exist_ok=True)
    for file_name in file_name_list:
        polder_time = file_name.split('_')[2]
        date_year_month_day = polder_time.split('T')[0]  # 2008-06-14
        doy = datetime.strptime(date_year_month_day, '%Y-%m-%d').timetuple().tm_yday
        output_name = f"{polder_time}_multi_pixel_yuenanV3.csv"

        print("=" * 60, "\n大规模多像元气溶胶反演")
        print(f"区域: Lon({lon_min}, {lon_max}), Lat({lat_min}, {lat_max})")
        print(f"并行数: {NUM_PROCESSES}\n", "=" * 60)

        # 验证文件路径
        full_file_path = os.path.join(file_path, file_name)
        if not os.path.exists(full_file_path):
            print(f"错误: 找不到文件 {full_file_path}")
            return
        if not os.path.exists(prior_path):
            print(f"错误: 找不到先验文件 {prior_path}")
            return

        print("加载模型和缩放器...")
        try:
            model_dict = declare_multi_model(config.wl_list, base_dir)
            features_scaler = joblib.load(os.path.join(base_dir, 'scaler_features.pkl'))
            print("加载完成.")
        except Exception as e:
            print(f"模型加载失败: {e}")
            return




        print("正在从源文件读取整个区域的原始像元数据...")
        io_start = time()

        try:
            with h5py.File(os.path.join(file_path, file_name), 'r') as f:
                lons_all = f['Geolocation_Fields']['Longitude'][:]
                lats_all = f['Geolocation_Fields']['Latitude'][:]

                mask = (lons_all >= lon_min) & (lons_all <= lon_max) & \
                       (lats_all >= lat_min) & (lats_all <= lat_max)

                valid_indices = np.argwhere(mask)
                if valid_indices.shape[0] == 0:
                    print("错误: 指定区域内没有找到任何有效的POLDER像元。")
                    return

                coordinates_to_extract = [tuple(idx) for idx in valid_indices]

                # 使用正确的原始经纬度
                correct_lons = lons_all[mask]
                correct_lats = lats_all[mask]

        except Exception as e:
            print(f"读取HDF5文件失败: {e}")
            return

        # 一次性矢量化提取所有数据
        all_pixel_data_from_file = vectorized_extract_polder_data(
            file_path, file_name, coordinates_to_extract
        )

        # # 关键：用正确的经纬度覆盖可能被污染的经纬度
        # print(len(all_pixel_data_from_file['lon_list']))
        # if len(all_pixel_data_from_file['lon_list']) <=  len(correct_lons):
        #     all_pixel_data_from_file['lon_list'] = correct_lons
        #     all_pixel_data_from_file['lat_list'] = correct_lats
        # else:
        #     print("有效向元大于总向元？")
        #     return

        print(f"I/O 和数据提取完成，耗时: {time() - io_start:.2f} 秒。共找到 {len(correct_lons)} 个有效像元。")

        # ========== 创建数据批次用于并行处理 (替换旧的区域划分逻辑) ==========
        pixel_batches = create_pixel_batches(all_pixel_data_from_file, NUM_PROCESSES)
        if not pixel_batches:
            print("没有可处理的数据批次。")
            return

        print(f"所有像元数据已被分割成 {len(pixel_batches)} 个批次进行并行处理。")

        tasks = []
        for i, batch in enumerate(pixel_batches):
            tasks.append({
                'id': i,
                'pixel_batch': batch,  # 传递像元数据批次
                'prior_path': prior_path,
                'config': config,
                'model_dict': model_dict,
                'features_scaler': features_scaler,
                'polder_time': polder_time,
                'doy':doy
            })
        print(f"\n=== 开始并行处理 {len(tasks)} 个块 ===")

        all_results = []
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            future_to_block = {executor.submit(process_data_block, task): task['id'] for task in tasks}
            with tqdm(total=len(tasks), desc="Processing Blocks") as pbar:
                for future in as_completed(future_to_block):
                    block_id = future_to_block[future]
                    try:
                        block_results = future.result()
                        all_results.extend(block_results)
                        print(f"块 {block_id} 完成，得到 {len(block_results)} 个结果")
                    except Exception as e:
                        print(f"处理块 {block_id} 时发生严重错误: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        pbar.update(1)

        # ========== 5. 保存结果 ==========
        if not all_results:
            print("处理完成，但没有得到任何有效结果。")
        else:
            result_df = pd.DataFrame(all_results)
            result_file = os.path.join(output_path, output_name)
            result_df.to_csv(result_file, index=False)
            print(f"\n处理完成，共得到 {len(all_results)} 个像元的结果。")
            print(f"结果已保存至: {result_file}")

            print("开始预测AOD...")
            try:
                predict_AOD(result_file, base_dir)
                print("AOD预测完成.")
            except Exception as e:
                print(f"AOD预测失败: {e}")

    T_end = time()
    print(f"\n程序总运行时间: {(T_end - T_begin) / 60:.2f} 分钟")


if __name__ == "__main__":
    #后加入的先验功能 暂时定义在主函数中
    VC_MODEL_PATH = '/media/amers/SSD_part1/whx/ResNet_code/dynamic_prior/vc_model_new.pth'
    SCALER_X_VC_PATH = '/media/amers/SSD_part1/whx/ResNet_code/dynamic_prior/dnn_feature_scaler_vc_new.joblib'
    SCALER_Y_VC_PATH = '/media/amers/SSD_part1/whx/ResNet_code/dynamic_prior/dnn_target_scaler_vc_new.joblib'
    VC_DEFAULT = 0.02
    # 加载先验的神经网络模型
    try:
        # 加载先验模型
        input_dim_vc = 136
        output_dim_vc = 4
        vc_model = ResidualDNN(input_dim_vc, output_dim_vc, layers_config=[1024, 512, 256, 128]).to('cpu')
        vc_model.load_state_dict(torch.load(VC_MODEL_PATH, map_location='cpu'))
        vc_model.eval()  # 切换到评估模式
        # 加载Scalers
        scaler_x_vc = joblib.load(SCALER_X_VC_PATH)
        scaler_y_vc = joblib.load(SCALER_Y_VC_PATH)
        print("神经网络模型和Scalers加载成功！")
    except FileNotFoundError as e:
        print(f"错误：找不到模型或Scaler文件！请确保文件存在。 {e}")
        vc_model = None
        scaler_x_vc = None
        scaler_y_vc = None

    main()