# --- START OF FILE DPC_aeronet_retrieval_NNprior.py ---

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
import torch.nn as nn

# 设置环境
warnings.filterwarnings('ignore', category=UserWarning)

device = torch.device('cpu')

# 导入自定义模块
from retrieval.moudle.ResNet_RTModel_pytorch import DNNModel
import retrieval.moudle.moudle_OE as OEfunc
from retrieval.moudle.predict_forAOD import predict_AOD_AAOD_SSA as predict_AOD
from retrieval.moudle.DNNModel import DNNModel as AerosolDNNModel

# ==============================================================================
# 1. 定义气溶胶先验预测网络模型 
# ==============================================================================
class ResidualDNN(nn.Module):
    def __init__(self, input_dim, output_dim, layers_config=[256, 128, 64],
                 dropout_rate=0.2, l2_regularization=1e-4):
        super(ResidualDNN, self).__init__()
        layers_list = []; current_dim = input_dim
        for i, units in enumerate(layers_config):
            block = nn.Sequential(
                nn.Linear(current_dim, units), 
                nn.LayerNorm(units), 
                nn.LeakyReLU(0.2), 
                nn.Dropout(dropout_rate)
            )
            layers_list.append(block)
            is_residual = (i != 0 and current_dim == units)
            layers_list.append(is_residual)
            current_dim = units
        self.hidden_layers = nn.ModuleList([layer for layer in layers_list if isinstance(layer, nn.Module)])
        self.residual_flags = [flag for flag in layers_list if isinstance(flag, bool)]
        self.output_layer = nn.Linear(current_dim, output_dim)
        
    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x_prev_layer = x
            x = layer(x)
            if self.residual_flags[i]: 
                x = x + x_prev_layer
        x = self.output_layer(x)
        return x


@dataclass
class RetrievalConfig:
    """反演配置类，便于管理和修改参数"""

    wl_list: List[str] = field(default_factory=lambda: ["0.443", "0.49", "0.565", "0.67", "0.865"])
    has_polarization: Dict[str, bool] = field(default_factory=lambda: {
        "0.443": False, "0.49": True, "0.565": False,
        "0.67": True, "0.865": True
    })

    total_parameters: List[str] = field(default_factory=lambda: [
        "sza", "vza", "fis", "vc_BB", "vc_Urban", "vc_Ocean", "vc_Dust", "ALH",
        "iso_0.443", "iso_0.49", "iso_0.565", "iso_0.67", "iso_0.865",
        "k1", "k2", "BPDF", "o3", "h2o", "dem"
    ])

    state_vector_list: List[str] = field(default_factory=lambda: [
        "vc_BB", "vc_Urban", "vc_Ocean", "vc_Dust", "ALH",
        "iso_0.443", "iso_0.49", "iso_0.565", "iso_0.67", "iso_0.865", 
        "k1", "k2", "BPDF"
    ])

    nonstate_vector_list: List[str] = field(default_factory=lambda: [
        "sza", "vza", "fis", "o3", "h2o", "dem"
    ])

    sigma_I: float = 0.05 
    sigma_dolp: float = 0.015  
    sigma_NN_I: List[float] = field(default_factory=lambda: [0.001, 0.0011, 0.0008, 0.001, 0.0011])
    sigma_NN_DOLP: List[float] = field(default_factory=lambda: [0.0, 0.0013, 0.0, 0.001, 0.0012])

    # === 新增：AERONET 真值约束容差 ===
    sigma_aeronet_aod: float = 0.001  
    sigma_aeronet_ssa: float = 0.001

    optimization_method: str = 'trf'  
    max_iterations: int = 30  
    xtol: float = 1e-7  
    ftol: float = 1e-7  
    gtol: float = 1e-7  

    # === 修改：关闭全局L2正则化，避免气溶胶被强行拉向0 ===
    use_regularization: bool = False  
    regularization_weight: float = 0.001

    use_prior: bool = True
    prior_weight: float = 0.1

    prior_type: Dict[str, str] = field(default_factory=lambda: {
        "vc_BB": "model",  
        "vc_Urban": "model",  
        "vc_Ocean": "model",  
        "vc_Dust": "model",  
        "ALH": "model",
        "iso": "climatology",
        "k1": "climatology",
        "k2": "climatology",
        "BPDF": "model"
    })

    prior_sigma: Dict[str, float] = field(default_factory=lambda: {
        "vc_BB": 0.1,  
        "vc_Urban": 0.1,  
        "vc_Ocean": 0.1,  
        "vc_Dust": 0.1,  
        "ALH": 0.5,  
        "iso": 0.1,  
        "k1": 0.1,  
        "k2": 0.1,  
        "BPDF": 0.3  
    })

    prior_weight_factor: Dict[str, float] = field(default_factory=lambda: {
        "observation": 0.6,  
        "climatology": 1.0,  
        "guess": 0.000000000001,  
        "model": 0.5  
    })

    def __post_init__(self):
        self.K = len(self.state_vector_list)
        self._init_bounds()
        self._init_state()
        self._init_obs_count()

    def _init_bounds(self):
        # === 修改：气溶胶下界改为 0.001，避免卡死在绝对0 ===
        self.state_bounds = [
            (0.00001, 1.5),  # vc_BB
            (0.00001, 1.5),  # vc_Urban
            (0.00001, 1.5),  # vc_Ocean
            (0.00001, 2),  # vc_Dust
            (0.5, 7),      # ALH
            (0.001, 0.8),  # iso_0.443
            (0.001, 0.8),  # iso_0.490
            (0.001, 0.8),  # iso_0.565
            (0.001, 0.8),  # iso_0.67
            (0.005, 0.8),  # iso_0.865
            (0.01, 2),     # k1
            (0.01, 2),     # k2
            (0.5, 8)       # BPDF
        ]

    def _init_state(self):
        self.init_state = np.array([
            0.1, 0.1, 0.1, 0.1, 3.5, 
            0.05, 0.06, 0.1, 0.12, 0.3, 
            0.6, 0.4, 4 
        ])

    def _init_obs_count(self):
        self.obs_count_per_wl = {
            wl: 2 if self.has_polarization[wl] else 1
            for wl in self.wl_list
        }


class ImprovedOptimizer:
    def __init__(self, config: RetrievalConfig):
        self.config = config

    def optimize_with_multistart(self, data, r_obs, model_dict, features_scaler,
                                 n_starts: int = 3) -> Tuple[np.ndarray, float]:
        best_state = None
        best_cost = np.inf

        for i in range(n_starts):
            if i == 0:
                init_state = self.config.init_state
            else:
                init_state = self._generate_random_state()

            result = self._optimize_single(data, r_obs, model_dict, features_scaler, init_state)
            final_cost, _, _ = current_total_cost_function_multi(
                result.x, data, r_obs, model_dict, features_scaler, self.config
            )

            if final_cost < best_cost:
                best_cost = final_cost
                best_state = result.x

        return best_state, best_cost

    def optimize_with_adaptive_weights(self, data, r_obs, model_dict, features_scaler,
                                       n_starts: int = 2) -> Tuple[np.ndarray, float, Dict]:
        import copy
        adaptive_config = copy.deepcopy(self.config)

        n_angles = r_obs.shape[0]
        n_obs_per_angle = r_obs.shape[1]
        has_pol = n_obs_per_angle > len(self.config.wl_list)  

        if n_angles < 5:
            adaptive_config.prior_weight *= 1.5
        elif n_angles > 10 and has_pol:
            adaptive_config.prior_weight *= 1

        adaptive_optimizer = ImprovedOptimizer(adaptive_config)
        best_state, best_cost = adaptive_optimizer.optimize_with_multistart(
            data, r_obs, model_dict, features_scaler, n_starts
        )

        diagnostics = {
            'n_angles': n_angles,
            'has_polarization': has_pol,
            'adapted_prior_weight': adaptive_config.prior_weight
        }

        return best_state, best_cost, diagnostics

    def _generate_random_state(self) -> np.ndarray:
        state = np.zeros(self.config.K)
        for i, (low, high) in enumerate(self.config.state_bounds):
            if low > 0 and high / low < 1000:  
                state[i] = np.exp(np.random.uniform(np.log(low), np.log(high)))
            else:
                state[i] = np.random.uniform(low, high)
        return state

    def check_feasibility(self, state_vector: np.ndarray) -> Tuple[bool, List[str]]:
        is_feasible = True
        violations = []
        for i, (value, (low, high)) in enumerate(zip(state_vector, self.config.state_bounds)):
            if value < low or value > high:
                is_feasible = False
                param_name = self.config.state_vector_list[i]
                violations.append(f"{param_name}: {value:.4f} not in [{low:.4f}, {high:.4f}]")
        return is_feasible, violations

    def _optimize_single(self, data, r_obs, model_dict, features_scaler, init_state) -> object:
        is_feasible, violations = self.check_feasibility(init_state)
        if not is_feasible:
            print(f"警告：初始值不可行！违反约束：{violations}")
            init_state = np.array([
                np.clip(init_state[i], self.config.state_bounds[i][0], self.config.state_bounds[i][1])
                for i in range(len(init_state))
            ])

        n_total = r_obs.size
        weights_np = compute_weights_improved(r_obs, self.config)
        sqrt_weights = np.sqrt(weights_np)

        prior_state = init_state.copy()
        prior_weights = self._get_prior_weights()

        # === 新增：解析 AOD/SSA 物理模型与真值 ===
        model_aerosol, scaler_x_aerosol, scaler_y_aerosol = model_dict['aerosol_prop']
        truth_aod = getattr(self.config, 'current_truth_aod', -999.0)
        truth_ssa = getattr(self.config, 'current_truth_ssa', -999.0)
        
        valid_aod = pd.notna(truth_aod) and truth_aod > 0
        valid_ssa = pd.notna(truth_ssa) and truth_ssa > 0

        def residual_function(sv):
            _, r_sim, _ = current_total_cost_function_multi(
                sv, data, r_obs, model_dict, features_scaler, self.config
            )
            diff = r_obs - r_sim
            residuals = (1.0 / np.sqrt(n_total)) * (sqrt_weights * diff).ravel()

            if self.config.use_prior:
                prior_residuals = self.config.prior_weight * prior_weights * (sv - prior_state)
                residuals = np.concatenate([residuals, prior_residuals])

            if self.config.use_regularization:
                reg_residuals = self.config.regularization_weight * sv
                residuals = np.concatenate([residuals, reg_residuals])

            # === 新增：AERONET 物理宏观约束 (AOD / SSA) ===
            if valid_aod or valid_ssa:
                current_vc = sv[0:4].reshape(1, -1)
                x_scaled_aerosol = scaler_x_aerosol.transform(current_vc).astype(np.float32)
                x_tensor = torch.from_numpy(x_scaled_aerosol).to(device)
                
                with torch.no_grad():
                    y_pred_scaled = model_aerosol(x_tensor).cpu().numpy()
                
                y_pred = scaler_y_aerosol.inverse_transform(y_pred_scaled)[0]
                pred_aod, pred_ssa = y_pred[0], y_pred[1]

                if valid_aod:
                    res_aod = np.array([(pred_aod - truth_aod) / self.config.sigma_aeronet_aod])
                    residuals = np.concatenate([residuals, res_aod])
                
                if valid_ssa:
                    res_ssa = np.array([(pred_ssa - truth_ssa) / self.config.sigma_aeronet_ssa])
                    residuals = np.concatenate([residuals, res_ssa])

            return residuals

        def jac_function(sv):
            _, _, jacobian = current_total_cost_function_multi(
                sv, data, r_obs, model_dict, features_scaler, self.config
            )
            J_val = -(sqrt_weights / np.sqrt(n_total)).reshape(-1, 1) * \
                    jacobian.reshape(-1, jacobian.shape[-1])

            if self.config.use_prior:
                prior_jac = self.config.prior_weight * np.diag(prior_weights)
                J_val = np.vstack([J_val, prior_jac])

            if self.config.use_regularization:
                reg_jac = self.config.regularization_weight * np.eye(self.config.K)
                J_val = np.vstack([J_val, reg_jac])

            # === 新增：AERONET 物理宏观约束的雅可比 (AOD / SSA) ===
            if valid_aod or valid_ssa:
                current_vc = sv[0:4].reshape(1, -1)
                x_scaled_aerosol = scaler_x_aerosol.transform(current_vc).astype(np.float32)
                
                x_tensor = torch.tensor(x_scaled_aerosol, requires_grad=True, device=device)
                
                def aod_model_func(x):
                    return model_aerosol(x)
                
                jac_tensor = torch.autograd.functional.jacobian(aod_model_func, x_tensor)
                jac_net = jac_tensor[0, :, 0, :].detach().cpu().numpy() 
                
                s_y = scaler_y_aerosol.scale_.astype(np.float64)  
                s_x = scaler_x_aerosol.scale_.astype(np.float64)  
                
                jac_actual = (jac_net * s_y.reshape(-1, 1)) / s_x.reshape(1, -1)
                
                full_jac_row_aod = np.zeros(self.config.K)
                full_jac_row_ssa = np.zeros(self.config.K)
                
                full_jac_row_aod[0:4] = jac_actual[0, :]  
                full_jac_row_ssa[0:4] = jac_actual[1, :]  
                
                if valid_aod:
                    J_val = np.vstack([J_val, full_jac_row_aod / self.config.sigma_aeronet_aod])
                
                if valid_ssa:
                    J_val = np.vstack([J_val, full_jac_row_ssa / self.config.sigma_aeronet_ssa])

            return J_val

        try:
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
                # === 修改：注释掉 x_scale='jac' ===
                # x_scale='jac',
                verbose=0
            )
        except Exception as e:
            print(f"优化失败: {e}")
            class FailedResult:
                def __init__(self, x):
                    self.x = x
                    self.success = False
                    self.message = str(e)
            result = FailedResult(init_state)

        return result

    def _get_prior_weights(self) -> np.ndarray:
        weights = np.zeros(self.config.K)
        for i, param in enumerate(self.config.state_vector_list):
            if param.startswith('vc_'):
                base_weight = 1.0 / self.config.prior_sigma.get('vc_BB', 1.0)
            elif param.startswith('iso_'):
                base_weight = 1.0 / self.config.prior_sigma.get('iso', 0.1)
            elif param in self.config.prior_sigma:
                base_weight = 1.0 / self.config.prior_sigma[param]
            else:
                base_weight = 1.0

            param_base = param.split('_')[0] if '_' in param else param
            prior_type = self.config.prior_type.get(param_base, 'guess')
            type_factor = self.config.prior_weight_factor.get(prior_type, 0.01)

            weights[i] = base_weight * type_factor
        return weights


def compute_weights_improved(r_obs: np.ndarray, config: RetrievalConfig) -> np.ndarray:
    weights = np.zeros_like(r_obs)
    current_col = 0
    for i, wl in enumerate(config.wl_list):
        I_obs = r_obs[:, current_col]
        sigma_NN_I = config.sigma_NN_I[i]
        sigma_I_dynamic = np.sqrt((config.sigma_I * I_obs) ** 2 + sigma_NN_I ** 2)
        min_sigma = 0.0001
        sigma_I_dynamic = np.maximum(sigma_I_dynamic, min_sigma)
        weights[:, current_col] = 1.0 / (sigma_I_dynamic ** 2)
        current_col += 1

        if config.has_polarization[wl]:
            sigma_NN_DOLP = config.sigma_NN_DOLP[i]
            if sigma_NN_DOLP > 0:  
                sigma_DOLP_dynamic = np.sqrt(config.sigma_dolp ** 2 + sigma_NN_DOLP ** 2)
                weights[:, current_col] = 1.0 / (sigma_DOLP_dynamic ** 2)
            else:
                weights[:, current_col] = 0  
            current_col += 1
    return weights


def parse_dpc_row_to_data_obs(row: pd.Series) -> np.ndarray:
    cal_coeffs = {
        '490P': 0.945797463, '565':  0.998798994,
        '670P': 1.015393182, '865P': 0.961277108
    }
    data_obs_list = []
    for ang_idx in range(12):  
        sza = row.get(f'sza_ang{ang_idx}')
        if pd.isna(sza) or sza <= 0:
            continue
            
        vza = row[f'vza_ang{ang_idx}']
        phi = row[f'phi_ang{ang_idx}']
        
        I443 = row[f'I443_ang{ang_idx}']
        I565 = row[f'I565_ang{ang_idx}']
        
        I490 = row.get(f'I490_ang{ang_idx}')
        Q490 = row.get(f'Q490_ang{ang_idx}')
        U490 = row.get(f'U490_ang{ang_idx}')
        if I490 is not None and not pd.isna(I490):
            I490 = I490 * cal_coeffs['490P']
            Q490 = Q490 * cal_coeffs['490P'] if Q490 is not None and not pd.isna(Q490) else 0.0
            U490 = U490 * cal_coeffs['490P'] if U490 is not None and not pd.isna(U490) else 0.0
        
        if I565 is not None and not pd.isna(I565):
            I565 = I565 * cal_coeffs['565']
        
        I670 = row.get(f'I670_ang{ang_idx}')
        Q670 = row.get(f'Q670_ang{ang_idx}')
        U670 = row.get(f'U670_ang{ang_idx}')
        if I670 is not None and not pd.isna(I670):
            I670 = I670 * cal_coeffs['670P']
            Q670 = Q670 * cal_coeffs['670P'] if Q670 is not None and not pd.isna(Q670) else 0.0
            U670 = U670 * cal_coeffs['670P'] if U670 is not None and not pd.isna(U670) else 0.0
        
        I865 = row.get(f'I865_ang{ang_idx}')
        Q865 = row.get(f'Q865_ang{ang_idx}')
        U865 = row.get(f'U865_ang{ang_idx}')
        if I865 is not None and not pd.isna(I865):
            I865 = I865 * cal_coeffs['865P']
            Q865 = Q865 * cal_coeffs['865P'] if Q865 is not None and not pd.isna(Q865) else 0.0
            U865 = U865 * cal_coeffs['865P'] if U865 is not None and not pd.isna(U865) else 0.0
        
        def calc_dolp(I, Q, U):
            if I is None or pd.isna(I) or I <= 1e-6: return 0.0
            Q_val = Q if Q is not None and not pd.isna(Q) else 0.0
            U_val = U if U is not None and not pd.isna(U) else 0.0
            return np.sqrt(Q_val**2 + U_val**2) / I
        
        DOLP490 = calc_dolp(I490, Q490, U490)
        DOLP670 = calc_dolp(I670, Q670, U670)
        DOLP865 = calc_dolp(I865, Q865, U865)
        
        angle_obs = [sza, vza, phi, I443, I490, DOLP490, I565, I670, DOLP670, I865, DOLP865]
        data_obs_list.append(angle_obs)
    
    return np.array(data_obs_list)


def build_vc_dnn_features(row: pd.Series, obs_time: pd.Timestamp) -> np.ndarray:
    cal_coeffs = {'490P': 1, '565': 1, '670P': 1, '865P': 1}
    features = [row['elev']]
    
    for ang_idx in range(12):
        sza = row.get(f'sza_ang{ang_idx}')
        if pd.isna(sza) or sza <= 0:
            features.extend([0.0] * 11)
            continue
            
        vza = row[f'vza_ang{ang_idx}']
        phi = row[f'phi_ang{ang_idx}']
        
        I443 = row[f'I443_ang{ang_idx}'] if pd.notna(row[f'I443_ang{ang_idx}']) else 0.0
        I565 = row[f'I565_ang{ang_idx}'] * cal_coeffs['565'] if pd.notna(row[f'I565_ang{ang_idx}']) else 0.0
        
        def get_calibrated_stokes(wl, coeff_key):
            I = row.get(f'I{wl}_ang{ang_idx}')
            Q = row.get(f'Q{wl}_ang{ang_idx}')
            U = row.get(f'U{wl}_ang{ang_idx}')
            I_cal = I * cal_coeffs[coeff_key] if pd.notna(I) else 0.0
            Q_cal = Q * cal_coeffs[coeff_key] if pd.notna(Q) else 0.0
            U_cal = U * cal_coeffs[coeff_key] if pd.notna(U) else 0.0
            DOLP = np.sqrt(Q_cal**2 + U_cal**2) / I_cal if I_cal > 1e-6 else 0.0
            return I_cal, DOLP

        I490, DOLP490 = get_calibrated_stokes('490', '490P')
        I670, DOLP670 = get_calibrated_stokes('670', '670P')
        I865, DOLP865 = get_calibrated_stokes('865', '865P')
        
        features.extend([sza, vza, phi, I443, I490, DOLP490, I565, I670, DOLP670, I865, DOLP865])
        
    return np.array(features)


def process_multi_angle_data_improved(data_obs, elev, lon, lat, prior_path, config: RetrievalConfig,
                                      priori_flag: bool, dynamic_prior_dict: Dict = None) -> Tuple[np.ndarray, np.ndarray, RetrievalConfig]:
    n_valid = data_obs.shape[0]

    n_obs_cols = sum(config.obs_count_per_wl.values())
    r_obs_matrix = np.zeros((n_valid, n_obs_cols))
    n_non_state = len(config.nonstate_vector_list)
    non_state_matrix = np.zeros((n_valid, n_non_state))

    non_state_matrix[:, 0:3] = data_obs[:, 0:3]

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

    try:
        iso_443, iso_490, iso_565, iso_670, iso_865, iso_1020, k1, k2, BPDF, ALH = \
            OEfunc.extract_prior_data(lon, lat, prior_path)

        prior_values = {
            'iso_0.443': np.clip(iso_443, 0.001, 0.8),
            'iso_0.49': np.clip(iso_490, 0.001, 0.8),
            'iso_0.565': np.clip(iso_565, 0.005, 0.8),
            'iso_0.67': np.clip(iso_670, 0.005, 0.8),
            'iso_0.865': np.clip(iso_865, 0.005, 0.8),
            'k1': np.clip(k1, 0.01, 2),
            'k2': np.clip(k2, 0.01, 2),
            'BPDF': np.clip(BPDF, 0.5, 8),
            'ALH': np.clip(ALH, 0.5, 8)
        }
    except Exception as e:
        print(f"先验数据获取失败: {e}，使用默认值")
        prior_values = {}  

    if dynamic_prior_dict:
        prior_values.update(dynamic_prior_dict)

    for i, param in enumerate(config.nonstate_vector_list[3:], 3):
        if param == 'dem':
            non_state_matrix[:, i] = elev
        elif param == 'o3':
            non_state_matrix[:, i] = 350.5  
        elif param == 'h2o':
            non_state_matrix[:, i] = 1.1  
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


def update_config_with_prior(config: RetrievalConfig, prior_values: Dict[str, float]) -> RetrievalConfig:
    import copy
    new_config = copy.deepcopy(config)

    for i, param in enumerate(new_config.state_vector_list):
        if param in prior_values:
            value = prior_values[param]

            if param.startswith('iso_'): margin = 0.1
            elif param == 'k1': margin = 0.3
            elif param == 'k2': margin = 0.2
            elif param == 'BPDF': margin = 2
            elif param == 'ALH': margin = 1
            elif param.startswith('vc_'): 
                # === 修改：动态调整边界范围，并更新先验类型 ===
                margin = max(value * 10, 0.2)
                new_config.prior_type[param] = "model"
                new_config.prior_type[param.split('_')[0]] = "model" 
                new_config.prior_sigma[param] = 0.5
            else:
                margin = 0.5

            original_lower, original_upper = new_config.state_bounds[i]
            lower = max(value - margin, original_lower)
            upper = min(value + margin, original_upper)

            if lower >= upper: 
                lower, upper = original_lower, original_upper

            init_value = np.clip(value, lower, upper)

            new_config.init_state[i] = init_value
            new_config.state_bounds[i] = (lower, upper)

    for i in range(len(new_config.init_state)):
        lower, upper = new_config.state_bounds[i]
        original_val = new_config.init_state[i] 
        
        if original_val < lower or original_val > upper:
            clipped_val = np.clip(original_val, lower, upper)
            new_config.init_state[i] = clipped_val
            
            param_name = new_config.state_vector_list[i]
            print(f"⚠️ [边界裁剪] 参数 '{param_name:<10}' | "
                  f"原初始值: {original_val:>7.4f} -> "
                  f"合法范围: [{lower:>7.4f}, {upper:>7.4f}] -> "
                  f"最终裁定值: {clipped_val:>7.4f}")

    return new_config


def quality_check(optimized_state: np.ndarray, final_cost: float,
                  config: RetrievalConfig) -> Dict[str, any]:
    quality = {
        'converged': final_cost < 0.05,  
        'final_cost': final_cost,
        'vc_total': None,
        'quality_flag': 0,  
        'warnings': []
    }

    vc_total = sum(optimized_state[i] for i in range(4))  
    quality['vc_total'] = vc_total

    if final_cost > 0.1:
        quality['quality_flag'] = 2
        quality['warnings'].append('High cost function value')
    elif final_cost > 0.001:
        quality['quality_flag'] = 1
        quality['warnings'].append('Moderate cost function value')

    if quality['vc_total'] > 3.0:
        quality['warnings'].append('Unusually high AOD')
        quality['quality_flag'] = max(quality['quality_flag'], 1)

    alh_idx = config.state_vector_list.index('ALH')
    if optimized_state[alh_idx] > 10.0:
        quality['warnings'].append('High aerosol layer height')

    return quality


def process_single_pixel_improved(pixel_index: int, data_obs: np.ndarray,
                                  elev: float, lon: float, lat: float, prior_path,
                                  config: RetrievalConfig, model_dict: Dict,
                                  features_scaler, priori_flag: bool,
                                  dynamic_prior_dict: Dict = None) -> Dict:
    try:
        # ======= 挂载 AERONET 真值到本次像元处理的专属 config 副本上 =======
        config.current_truth_aod = dynamic_prior_dict.get('truth_aod', -999.0) if dynamic_prior_dict else -999.0
        config.current_truth_ssa = dynamic_prior_dict.get('truth_ssa', -999.0) if dynamic_prior_dict else -999.0
        
        r_obs_matrix, non_state_matrix, new_config = process_multi_angle_data_improved(
            data_obs, elev, lon, lat, prior_path, config, priori_flag, dynamic_prior_dict
        )

        if r_obs_matrix.shape[0] == 0:
            return {'pixel_index': pixel_index, 'error': '没有有效观测数据'}

        n_rows = non_state_matrix.shape[0]
        total_length = 3 + len(new_config.init_state) + (non_state_matrix.shape[1] - 3)
        data_matrix = np.zeros((n_rows, total_length))
        data_matrix[:, :3] = non_state_matrix[:, :3]
        data_matrix[:, 3:3 + len(new_config.init_state)] = np.tile(new_config.init_state, (n_rows, 1))
        data_matrix[:, 3 + len(new_config.init_state):] = non_state_matrix[:, 3:]

        optimizer = ImprovedOptimizer(new_config)
        optimized_state, final_cost, diagnostics = optimizer.optimize_with_adaptive_weights(
            data_matrix, r_obs_matrix, model_dict, features_scaler, n_starts=1
        )

        quality_info = quality_check(optimized_state, final_cost, new_config)
        quality_info['diagnostics'] = diagnostics  

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
                                      elev_list: List, lon_list: List, lat_list: List, 
                                      times_list: List, site_names: List, prior_path_list: List,
                                      dynamic_prior_list: List[Dict], 
                                      output_path: str, output_name: str,
                                      model_dict: Dict, features_scaler,
                                      priori_flag: bool = True, n_jobs: int = -1) -> List[Dict]:
    n_pixels = len(data_obs_list)
    print(f"开始处理 {n_pixels} 个像元...")

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_single_pixel_improved)(
            i, data_obs_list[i], elev_list[i], lon_list[i], lat_list[i], prior_path_list[i],
            config, model_dict, features_scaler, priori_flag, 
            dynamic_prior_list[i] 
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

    result_df.to_csv(result_file, index=False)
    print(f"结果保存至: {result_file}")

    return all_results


def declare_multi_model(wl_list: List[str], model_dir: str, device: str = 'cpu') -> Dict:
    model_dict = {}
    for wl in wl_list:
        model_filename = os.path.join(model_dir, f"dnn_model_{wl}.pth")
        scaler_filename = os.path.join(model_dir, f"scaler_y_{wl}.pkl")
        try:
            model = DNNModel.load_model(model_filename, device=device)
            model.eval()
            target_scaler = joblib.load(scaler_filename)
            model_dict[wl] = (model, target_scaler)
        except FileNotFoundError:
            print(f"[错误] 找不到PyTorch模型文件: {model_filename}")
            raise
    return model_dict


def update_state_in_data(data: np.ndarray, state_vector: np.ndarray,
                         config: RetrievalConfig) -> np.ndarray:
    n = data.shape[0]
    m = len(config.wl_list)
    geom_params = data[:, :3]
    state_len = len(config.state_vector_list)
    other_params = data[:, 3 + state_len:]

    basic_params = state_vector[:5]
    remaining_params = state_vector[-3:]
    brdf_values = np.array([state_vector[5 + j] for j in range(m)])

    geom_rep = np.repeat(geom_params, m, axis=0)
    other_rep = np.repeat(other_params, m, axis=0)
    basic_rep = np.tile(basic_params, (n * m, 1))
    remaining_rep = np.tile(remaining_params, (n * m, 1))
    brdf_rep = np.tile(brdf_values, n).reshape(-1, 1)

    data_updated = np.concatenate([geom_rep, basic_rep, brdf_rep, remaining_rep, other_rep], axis=1)
    return data_updated


def predict_multi_wavelength(model_dict: Dict, data_scaled: np.ndarray,
                             config: RetrievalConfig, features_scaler) -> Tuple[np.ndarray, np.ndarray]:
    n_angles = data_scaled.shape[0] // len(config.wl_list)
    K = len(config.state_vector_list)
    total_cols = sum(2 if config.has_polarization[wl] else 1 for wl in config.wl_list)

    r_sim_all = np.zeros((n_angles, total_cols))
    Kp_all = np.zeros((n_angles, total_cols, K))

    current_col = 0
    for i, wl in enumerate(config.wl_list):
        model, target_scaler = model_dict[wl]

        wl_indices = list(range(i, data_scaled.shape[0], len(config.wl_list)))
        data_for_model = data_scaled[wl_indices].astype(np.float32)

        input_tensor = torch.from_numpy(data_for_model).float()

        with torch.no_grad():
            y_pred_scaled_tensor = model(input_tensor)

        y_pred_scaled = y_pred_scaled_tensor.numpy()
        y_pred = target_scaler.inverse_transform(y_pred_scaled).astype(np.float64)

        def model_func(x):
            return model(x)

        jacobian_tensor = torch.autograd.functional.jacobian(model_func, input_tensor, vectorize=True)
        jacobian_full_tensor = torch.diagonal(jacobian_tensor, offset=0, dim1=0, dim2=2).permute(2, 0, 1)
        jacobian_full = jacobian_full_tensor.detach().numpy()

        if config.has_polarization[wl]:
            brdf_start_idx = 8
            r_sim_all[:, current_col:current_col + 2] = y_pred

            s_y = target_scaler.scale_.astype(np.float64)
            jacobian_output_adjusted = jacobian_full * s_y.reshape(1, -1, 1)
            s_x = features_scaler.scale_.astype(np.float64)
            jacobian_actual = jacobian_output_adjusted / s_x.reshape(1, 1, -1)

            full_jacobian = np.zeros((n_angles, 2, K))
            full_jacobian[:, :, :5] = jacobian_actual[:, :, 3:brdf_start_idx]
            full_jacobian[:, :, 5 + i] = jacobian_actual[:, :, brdf_start_idx]
            full_jacobian[:, :, 5 + len(config.wl_list):] = jacobian_actual[:, :, brdf_start_idx + 1:brdf_start_idx + 4]

            Kp_all[:, current_col:current_col + 2, :] = full_jacobian
            current_col += 2
        else:
            brdf_start_idx = 8
            r_sim_all[:, current_col] = y_pred[:, 0]

            s_y = target_scaler.scale_[0].astype(np.float64)
            jacobian_output_adjusted = jacobian_full[:, 0, :] * s_y
            s_x = features_scaler.scale_.astype(np.float64)
            jacobian_actual = jacobian_output_adjusted / s_x

            full_jacobian = np.zeros((n_angles, K))
            full_jacobian[:, :5] = jacobian_actual[:, 3:brdf_start_idx]
            full_jacobian[:, 5 + i] = jacobian_actual[:, brdf_start_idx]
            full_jacobian[:, 5 + len(config.wl_list):] = jacobian_actual[:, brdf_start_idx + 1:brdf_start_idx + 4]

            Kp_all[:, current_col, :] = full_jacobian
            current_col += 1

    return r_sim_all, Kp_all


def current_total_cost_function_multi(state_vector: np.ndarray, data: np.ndarray,
                                      r_obs: np.ndarray, model_dict: Dict,
                                      features_scaler, config: RetrievalConfig) -> Tuple[float, np.ndarray, np.ndarray]:
    data_updated = update_state_in_data(data, state_vector, config)
    data_scaled = features_scaler.transform(data_updated)
    r_sim, Kp = predict_multi_wavelength(model_dict, data_scaled, config, features_scaler)

    diff = r_obs - r_sim
    n_total = r_obs.size
    weights = compute_weights_improved(r_obs, config)

    weighted_diff = weights * (diff ** 2)
    cost = np.sum(weighted_diff) / n_total

    return cost, r_sim, Kp


if __name__ == "__main__":
    T_begin = time()
    
    #DPC_CSV_PATH = '/media/amers/SSD_part1/whx/ResNet_forDPC/DPC_Aeronet_2025_withPrior.csv'
    DPC_CSV_PATH = '/media/amers/SSD_part1/whx/ResNet_forDPC/DPC_Aeronet_2025_withPrior.csv'
    try:
        dpc_df = pd.read_csv(DPC_CSV_PATH)
        print(f"成功读取 DPC CSV 匹配文件，共 {len(dpc_df)} 条记录。")
    except FileNotFoundError:
        print(f"[错误] 找不到 DPC CSV 文件: '{DPC_CSV_PATH}'")
        exit()

    config = RetrievalConfig()

    output_path = r'/media/amers/SSD_part1/whx/ResNet_forDPC/'
    os.makedirs(output_path, exist_ok=True)
    output_name = "dpc_retrieval_results_withprior3.csv"

    # ==============================================================================
    # 加载 正向辐射传输模型(DNN)
    # ==============================================================================
    print("\n===== 加载多波段正向 DNN 模型 =====")
    base_dir = '/media/amers/SSD_part1/whx/ResNet_code/forward/V1/resnet_param/'
    model_dict = declare_multi_model(config.wl_list, base_dir, device=device)
    features_scaler = joblib.load(os.path.join(base_dir, 'scaler_features.pkl'))

    # ==============================================================================
    # 加载 气溶胶体积浓度预测模型(VC Model)
    # ==============================================================================
    print("\n===== 加载动态先验 VC DNN 模型 =====")
    VC_MODEL_PATH = '/media/amers/SSD_part1/whx/ResNet_forDPC/dynamic_prior/vc_model_DPC.pth'
    SCALER_X_VC_PATH = '/media/amers/SSD_part1/whx/ResNet_forDPC/dynamic_prior/dnn_feature_scaler_vc_DPC.joblib'
    SCALER_Y_VC_PATH = '/media/amers/SSD_part1/whx/ResNet_forDPC/dynamic_prior/dnn_target_scaler_vc_DPC.joblib'
    
    vc_model = ResidualDNN(133, 4, layers_config=[1024, 512, 256, 128]).to(device)   
    vc_model.load_state_dict(torch.load(VC_MODEL_PATH, map_location=device))
    vc_model.eval()
    scaler_x_vc = joblib.load(SCALER_X_VC_PATH)
    scaler_y_vc = joblib.load(SCALER_Y_VC_PATH)

    # ==============================================================================
    # 加载 AOD/SSA 物理约束 DNN 模型
    # ==============================================================================
    print("\n===== 加载 AOD/SSA 物理约束 DNN 模型 =====")
    AEROSOL_MODEL_DIR = base_dir 
    model_aerosol = AerosolDNNModel.load_model(os.path.join(AEROSOL_MODEL_DIR, 'dnn_model_aerosol.pth'))
    model_aerosol.to(device)
    model_aerosol.eval()
    scaler_x_aerosol = joblib.load(os.path.join(AEROSOL_MODEL_DIR, 'scaler_features_aerosol.pkl'))
    scaler_y_aerosol = joblib.load(os.path.join(AEROSOL_MODEL_DIR, 'scaler_y_aerosol.pkl'))
    
    # 挂载到 model_dict 中供优化器使用
    model_dict['aerosol_prop'] = (model_aerosol, scaler_x_aerosol, scaler_y_aerosol)

    # 初始化批处理列表
    data_obs_list, elev_list, lon_list, lat_list = [], [], [], []
    times_list, site_names, prior_path_list = [], [], []
    dynamic_prior_list = [] 

    print("\n===== 开始解析 DPC 数据并预测动态先验 =====")
    for index, row in tqdm(dpc_df.iterrows(), total=len(dpc_df), desc="Parsing & Predicting"):
        try:
            obs_time = pd.to_datetime(row['sat_time_utc'])
            month_str = f"{obs_time.month:02d}"

            prior_path = f'/media/amers/WHX/NNOE_POLDER/POLDER_data/priori_data/GRASP_priori_data_{month_str}.nc'
            
            data_obs = parse_dpc_row_to_data_obs(row)
            
            if len(data_obs) > 0:
                vc_features = build_vc_dnn_features(row, obs_time)
                
                X_scaled = scaler_x_vc.transform(vc_features.reshape(1, -1))
                X_tensor = torch.from_numpy(X_scaled).float().to(device)
                with torch.no_grad():
                    y_pred_scaled = vc_model(X_tensor).cpu().numpy()
                y_pred = scaler_y_vc.inverse_transform(y_pred_scaled)[0]
                
                y_pred[y_pred < 0] = 0.0 
                
                # === 修改：不再强行给 0.15，而是给一个合理的物理极小值 ===
                vc_bb    = max(y_pred[0], 0.005)
                vc_urban = max(y_pred[1], 0.005)
                vc_ocean = max(y_pred[2], 0.005)
                vc_dust  = max(y_pred[3], 0.005)

                # === 新增：提取 AERONET 真值，注意替换为你 CSV 中的实际列名 ===
                truth_aod = row.get('Pred_AOD_550nm', -999.0)
                truth_ssa = row.get('Pred_SSA_550nm', -999.0)

                dynamic_vc_prior = {
                    'vc_BB': vc_bb,
                    'vc_Urban': vc_urban,
                    'vc_Ocean': vc_ocean,
                    'vc_Dust': vc_dust,
                    'truth_aod': truth_aod,
                    'truth_ssa': truth_ssa
                }

                data_obs_list.append(data_obs)
                elev_list.append(row['elev']) 
                lon_list.append(row['lon'])
                lat_list.append(row['lat'])
                times_list.append(row['sat_time_utc'])
                site_names.append(row['site_name'])
                prior_path_list.append(prior_path)
                dynamic_prior_list.append(dynamic_vc_prior) 
                
        except Exception as e:
            print(f"解析第 {index} 行数据时发生错误: {e}")
            continue

    print(f"数据解析与先验预测完成，共提取 {len(data_obs_list)} 个有效像元等待反演。")

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
            dynamic_prior_list=dynamic_prior_list, 
            output_path=output_path, 
            output_name=output_name, 
            model_dict=model_dict, 
            features_scaler=features_scaler, 
            priori_flag=True, 
            n_jobs=-1
        )
        
        predict_AOD(os.path.join(output_path, output_name), base_dir)
    else:
        print("未提取到任何有效数据。")

    T_end = time()
    print(f"\n程序总运行时间: {(T_end - T_begin) / 60:.2f} 分钟")
    print('完成')