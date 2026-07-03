"""
DPC_L1_retrieval.py
====================
从 DPC L1 HDF5 文件直接进行气溶胶最优估计反演（OE Retrieval）。

【架构说明】
  参考 POLDER 多像元并行代码的成功模式：
  - process_pixel_batch() 是纯函数，所有依赖通过参数传入，可被 pickle
  - ProcessPoolExecutor 外层并行，每个进程处理一批像元
  - OE 核心（least_squares）在子进程内单线程串行执行，无嵌套并行
  - 主进程负责 H5 读取、掩膜、特征提取、批次划分

【数据流】
  L1 H5 → 读取全块 → 掩膜过滤 → 批量 VC DNN 推理
  → 划分批次 → ProcessPoolExecutor → 每批串行 OE → 汇总写 NC

【修复说明：为什么之前卡死在 25%】
  旧版本将 process_single_pixel_improved 包在类方法中，
  ProcessPoolExecutor 序列化（pickle）整个类实例时失败或卡死。
  本版本将 process_pixel_batch 定义为模块级纯函数，
  所有依赖（model_dict / config / data_obs）均通过参数传入，
  与参考代码结构完全一致，解决死锁问题。
"""

# 必须在所有 import 之前设置，防止子进程中 numpy/mkl 抢占线程造成竞争
import os
os.environ['OMP_NUM_THREADS']        = '1'
os.environ['OPENBLAS_NUM_THREADS']   = '1'
os.environ['MKL_NUM_THREADS']        = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS']    = '1'

import re
import glob
import warnings
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import netCDF4 as nc
import h5py
import torch
import joblib
from tqdm import tqdm
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

# ==============================================================================
# 自定义模块
# ==============================================================================
from retrieval.moudle.ResNet_RTModel_pytorch import DNNModel
from retrieval.moudle.DNNModel import DNNModel as AerosolDNNModel

from DPC_aeronet_retrieval_NNprior_AOD_SSAconstrain import (
    RetrievalConfig,
    ResidualDNN,
    process_single_pixel_improved,
)

# ==============================================================================
# 常量
# ==============================================================================
SCALE_REF  = 0.0001
SCALE_ANG  = 0.01
NODATA_REF = 32767
LAND_FLAG  = 100

CAL_COEFFS = {
    '490': 0.945797463,
    '565': 0.998798994,
    '670': 1.015393182,
    '865': 0.961277108,
}

# 角度索引 2~13，共 12 个（与 Transformer 推理保持一致）
ANGLE_INDICES = list(range(2, 14))
N_ANGLES      = len(ANGLE_INDICES)    # 12
VC_FEAT_DIM   = 1 + N_ANGLES * 11    # 133


# ==============================================================================
# 工具函数（无状态，主进程和子进程均可安全调用）
# ==============================================================================

def find_h5(folder: str, band: str) -> Optional[str]:
    fs = glob.glob(os.path.join(folder, f'*B{band}.h5'))
    return fs[0] if fs else None


def calc_dolp(I: np.ndarray, Q: np.ndarray, U: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(I > 1e-6, np.sqrt(Q**2 + U**2) / I, 0.0)


def build_mask(lat_block, lon_block, land_sea,
               i565_nadir, i490_nadir,
               lat_min, lat_max, lon_min, lon_max) -> np.ndarray:
    mask  = (land_sea == LAND_FLAG)
    mask &= (lat_block >= lat_min) & (lat_block <= lat_max)
    mask &= (lon_block >= lon_min) & (lon_block <= lon_max)
    mask &= (i565_nadir != NODATA_REF) & (i565_nadir > 0)
    mask &= (i490_nadir != NODATA_REF) & (i490_nadir > 0)
    ref565 = i565_nadir.astype(np.float64) * SCALE_REF * CAL_COEFFS['565']
    ref490 = i490_nadir.astype(np.float64) * SCALE_REF * CAL_COEFFS['490']
    mask &= ~((ref565 > 0.40) | (ref490 > 0.25))
    return mask


def infer_month(h5_file, l1_dir: str) -> int:
    for attr in ('DateTime', 'StartTime', 'date', 'Date'):
        val = h5_file.attrs.get(attr)
        if val is not None:
            try:
                s = val.decode() if isinstance(val, bytes) else str(val)
                return datetime.strptime(s[:8], '%Y%m%d').month
            except Exception:
                pass
    m = re.search(r'(\d{8})', os.path.basename(l1_dir.rstrip('/\\')))
    if m:
        try:
            return datetime.strptime(m.group(1), '%Y%m%d').month
        except ValueError:
            pass
    print("⚠️ 无法推断月份，默认使用 1 月先验。")
    return 1


def read_vc_prior_nc(nc_path: Optional[str],
                     lat_flat: np.ndarray,
                     lon_flat: np.ndarray) -> dict:
    """KDTree 最近邻从 Transformer 先验 NC 读取 AOD/SSA。"""
    nan_arr = lambda: np.full(len(lat_flat), np.nan, dtype=np.float32)
    result = {'aod': nan_arr(), 'ssa': nan_arr()}
    if nc_path is None or not os.path.exists(nc_path):
        return result
    with nc.Dataset(nc_path, 'r') as ds:
        nc_lat = np.array(ds.variables['latitude'][:])
        nc_lon = np.array(ds.variables['longitude'][:])
        av = ds.variables.get('AOD_550nm')
        sv = ds.variables.get('SSA_550nm')
        if av is None or sv is None:
            return result
        aod_f = np.array(av[:]).ravel()
        ssa_f = np.array(sv[:]).ravel()
    valid = np.isfinite(aod_f) & np.isfinite(ssa_f)
    if valid.sum() == 0:
        return result
    tree = cKDTree(np.c_[nc_lat.ravel()[valid], nc_lon.ravel()[valid]])
    dists, idxs = tree.query(np.c_[lat_flat, lon_flat], k=1)
    hit = dists <= 0.05          # 约 5 km 距离阈值
    result['aod'][hit] = aod_f[valid][idxs[hit]]
    result['ssa'][hit] = ssa_f[valid][idxs[hit]]
    return result


# ==============================================================================
# 像元特征提取（在主进程中调用）
# ==============================================================================

def build_data_obs_single(sza, vza, phi,
                           I443, I490, DOLP490,
                           I565, I670, DOLP670,
                           I865, DOLP865,
                           pix_idx: int) -> np.ndarray:
    """
    组装单像元观测矩阵，shape = (n_valid_angles, 11)。
    输入数组 shape = (n_ang_total, N_flat)。
    列：[sza, vza, phi, I443, I490, DOLP490, I565, I670, DOLP670, I865, DOLP865]
    """
    rows = []
    for ang in ANGLE_INDICES:
        sv = sza[ang, pix_idx]
        if sv <= 0 or not np.isfinite(sv):
            continue
        rows.append([sv,
                     vza[ang, pix_idx],  phi[ang, pix_idx],
                     I443[ang, pix_idx],
                     I490[ang, pix_idx], DOLP490[ang, pix_idx],
                     I565[ang, pix_idx],
                     I670[ang, pix_idx], DOLP670[ang, pix_idx],
                     I865[ang, pix_idx], DOLP865[ang, pix_idx]])
    return np.array(rows, dtype=np.float32) if rows else np.empty((0, 11), dtype=np.float32)


def build_vc_feat_single(elev_val, sza, vza, phi,
                          I443, I490, DOLP490,
                          I565, I670, DOLP670,
                          I865, DOLP865,
                          pix_idx: int) -> np.ndarray:
    """组装 VC DNN 输入特征向量（133 维 = 1 + 12×11）。"""
    feats = [elev_val]
    for ang in ANGLE_INDICES:
        sv = sza[ang, pix_idx]
        if sv <= 0 or not np.isfinite(sv):
            feats.extend([0.0] * 11)
            continue
        feats.extend([sv,
                      vza[ang, pix_idx],  phi[ang, pix_idx],
                      I443[ang, pix_idx],
                      I490[ang, pix_idx], DOLP490[ang, pix_idx],
                      I565[ang, pix_idx],
                      I670[ang, pix_idx], DOLP670[ang, pix_idx],
                      I865[ang, pix_idx], DOLP865[ang, pix_idx]])
    return np.array(feats, dtype=np.float32)


# ==============================================================================
# ★ 子进程工作函数（模块级纯函数，可被 pickle，无嵌套并行）★
# ==============================================================================

def process_pixel_batch(batch_info: dict) -> List[dict]:
    """
    处理一批像元的完整反演流程。

    由 ProcessPoolExecutor 在独立子进程中调用。
    ★ 关键设计：模块级纯函数，所有依赖通过 batch_info 字典传入，
      不引用任何类实例，pickle 安全，无嵌套进程池。

    batch_info 字段：
        batch_id        : int
        pixel_list      : list[dict]  每个 dict 含 data_obs/elev/lon/lat/
                                      flat_idx/dynamic_prior
        prior_nc_path   : str
        config          : RetrievalConfig
        model_dict      : dict
        features_scaler : sklearn scaler
    """
    batch_id        = batch_info['batch_id']
    pixel_list      = batch_info['pixel_list']
    prior_nc        = batch_info['prior_nc_path']
    config          = batch_info['config']
    model_dict      = batch_info['model_dict']
    features_scaler = batch_info['features_scaler']

    n = len(pixel_list)
    print(f"[批次 {batch_id:04d}] 开始处理 {n} 个像元...", flush=True)

    results = []
    for pix in pixel_list:
        try:
            res = process_single_pixel_improved(
                pixel_index        = pix['pixel_index'],
                data_obs           = pix['data_obs'],
                elev               = pix['elev'],
                lon                = pix['lon'],
                lat                = pix['lat'],
                prior_path         = prior_nc,
                config             = config,
                model_dict         = model_dict,
                features_scaler    = features_scaler,
                priori_flag        = True,
                dynamic_prior_dict = pix['dynamic_prior'],
            )
            if res.get('error'):
                results.append({
                    'flat_idx'       : pix['flat_idx'],
                    'optimized_state': None,
                    'final_cost'     : np.nan,
                    'quality_flag'   : 3,
                })
            else:
                qi = res.get('quality_info', {})
                results.append({
                    'flat_idx'       : pix['flat_idx'],
                    'optimized_state': res.get('optimized_state'),
                    'final_cost'     : res.get('final_cost', np.nan),
                    'quality_flag'   : qi.get('quality_flag', 3),
                })
        except Exception as e:
            results.append({
                'flat_idx'       : pix['flat_idx'],
                'optimized_state': None,
                'final_cost'     : np.nan,
                'quality_flag'   : 3,
            })

    success = sum(1 for r in results if r['optimized_state'] is not None)
    print(f"[批次 {batch_id:04d}] 完成：{success} / {n} 成功", flush=True)
    return results


# ==============================================================================
# 模型加载（主进程）
# ==============================================================================

def load_all_models(forward_model_dir, vc_model_path,
                    scaler_x_vc_path, scaler_y_vc_path,
                    config: RetrievalConfig):
    """加载所有 DNN 模型与 Scaler，返回相应对象。"""
    print("\n===== 加载多波段正向 DNN 模型 =====")
    model_dict = {}
    for wl in config.wl_list:
        m  = DNNModel.load_model(
            os.path.join(forward_model_dir, f'dnn_model_{wl}.pth'), device='cpu')
        m.eval()
        ts = joblib.load(os.path.join(forward_model_dir, f'scaler_y_{wl}.pkl'))
        model_dict[wl] = (m, ts)
    features_scaler = joblib.load(
        os.path.join(forward_model_dir, 'scaler_features.pkl'))

    print("===== 加载 VC-> AOD/SSA  DNN 模型 =====")
    am = AerosolDNNModel.load_model(
        os.path.join(forward_model_dir, 'dnn_model_aerosol.pth'))
    am.to('cpu').eval()
    sx_a = joblib.load(os.path.join(forward_model_dir, 'scaler_features_aerosol.pkl'))
    sy_a = joblib.load(os.path.join(forward_model_dir, 'scaler_y_aerosol.pkl'))
    model_dict['aerosol_prop'] = (am, sx_a, sy_a)

    print("===== 加载 VC 动态先验 DNN 模型 =====")
    vc_model = ResidualDNN(133, 4, layers_config=[1024, 512, 256, 128])
    vc_model.load_state_dict(torch.load(vc_model_path, map_location='cpu'))
    vc_model.eval()
    sx_vc = joblib.load(scaler_x_vc_path)
    sy_vc = joblib.load(scaler_y_vc_path)

    print("✅ 所有模型加载完成。\n")
    return model_dict, features_scaler, vc_model, sx_vc, sy_vc


# ==============================================================================
# 主处理函数
# ==============================================================================

# ==============================================================================
# 主处理函数
# ==============================================================================

def run_retrieval(
    l1_dir:            str,
    output_nc_path:    str,
    forward_model_dir: str,
    prior_nc_dir:      str,
    vc_model_path:     str,
    scaler_x_vc_path:  str,
    scaler_y_vc_path:  str,
    vc_prior_nc:       Optional[str] = None,
    lon_min: float = 70.0,
    lon_max: float = 140.0,
    lat_min: float = 7.0,
    lat_max: float = 55.0,
    block_size:  int = 512,
    n_processes: int = 8,
    batch_size:  int = 30,
):
    """
    主流程：
      1. 读取 L1 H5 → 掩膜 → 批量 VC 推理 → 组装像元任务列表
      2. 按 batch_size 切分 → ProcessPoolExecutor 并行 OE
      3. 汇总写出 NetCDF4 (包含 Prior Group 和 宏观光学参数预测)
    """
    t0 = time.time()

    # ── 1. 查找 H5 文件 ──────────────────────────────────────────────────────
    bands = ('443', '490', '565', '670', '865')
    files = {b: find_h5(l1_dir, b) for b in bands}
    files['geo'] = files['443']
    missing = [k for k, v in files.items() if v is None]
    if missing:
        raise FileNotFoundError(f"缺少波段 H5 文件: {missing}  目录: {l1_dir}")
    print(f"✅ 找到所有 H5 文件：{l1_dir}")

    # ── 2. 推断月份 ───────────────────────────────────────────────────────────
    with h5py.File(files['geo'], 'r') as f:
        H, W = f['Geolocation_Fields']['Latitude'].shape
        month = infer_month(f, l1_dir)
    prior_nc_path = os.path.join(prior_nc_dir, f'GRASP_priori_data_{month:02d}.nc')
    print(f"📅 月份：{month:02d}，GRASP 先验 NC：{prior_nc_path}")

    # ── 3. 加载模型 ───────────────────────────────────────────────────────────
    config = RetrievalConfig()
    model_dict, features_scaler, vc_model, sx_vc, sy_vc = load_all_models(
        forward_model_dir, vc_model_path,
        scaler_x_vc_path, scaler_y_vc_path, config)

    # ── 4. 初始化输出 NC ──────────────────────────────────────────────────────
    if os.path.exists(output_nc_path):
        os.remove(output_nc_path)
    nc_out = nc.Dataset(output_nc_path, 'w', format='NETCDF4')
    nc_out.createDimension('line', H)
    nc_out.createDimension('pixel', W)
    nc_out.title  = 'DPC L1 OE Aerosol Retrieval'
    nc_out.source = os.path.basename(l1_dir.rstrip('/\\'))

    lat_var = nc_out.createVariable('latitude',  'f4', ('line', 'pixel'), zlib=True)
    lon_var = nc_out.createVariable('longitude', 'f4', ('line', 'pixel'), zlib=True)
    lat_var.units = 'degrees_north'
    lon_var.units = 'degrees_east'

    # 根目录：反演状态量输出变量
    state_vars = {}
    for sv in config.state_vector_list:
        v = nc_out.createVariable(sv, 'f4', ('line', 'pixel'),
                                  zlib=True, complevel=4, fill_value=np.nan)
        v.long_name = sv
        state_vars[sv] = v
        
    cost_var = nc_out.createVariable('final_cost',   'f4', ('line', 'pixel'), zlib=True, fill_value=np.nan)
    flag_var = nc_out.createVariable('quality_flag', 'i2', ('line', 'pixel'), zlib=True, fill_value=-1)

    # 【新增】：创建 Prior Group，专门存储先验变量
    grp_prior = nc_out.createGroup('Prior')
    prior_vars = {}
    for sv in ['vc_BB', 'vc_Urban', 'vc_Ocean', 'vc_Dust', 'AOD_550nm', 'SSA_550nm']:
        v = grp_prior.createVariable(sv, 'f4', ('line', 'pixel'), zlib=True, fill_value=np.nan)
        prior_vars[sv] = v

# ── 5. 全图缓冲区（写回时用） ─────────────────────────────────────────────
    lat_buf      = np.full((H, W), np.nan, dtype=np.float32)
    lon_buf      = np.full((H, W), np.nan, dtype=np.float32)
    
    # 状态量与诊断量 Buffer
    sv_buf       = {sv: np.full(H * W, np.nan, dtype=np.float32) for sv in config.state_vector_list}
    cost_buf     = np.full(H * W, np.nan,  dtype=np.float32)
    flag_buf     = np.full(H * W, -1,      dtype=np.int16)
    
    # 先验量 Buffer
    prior_vc_buf = {sv: np.full(H * W, np.nan, dtype=np.float32) for sv in ['vc_BB', 'vc_Urban', 'vc_Ocean', 'vc_Dust']}
    aod_prior_buf = np.full(H * W, np.nan, dtype=np.float32)
    ssa_prior_buf = np.full(H * W, np.nan, dtype=np.float32)

# =========================================================================
    # 🌟 新增：BBox (Bounding Box) 极速优化探测
    # =========================================================================
    print("🔍 正在探测轨道有效边界 (BBox)...")
    h5s = {k: h5py.File(v, 'r') for k, v in files.items()}
    
    # 【修改】：使用经纬度来探测真实边界。DPC 的无效经纬度通常是 32767 或 99999
    # 我们直接寻找纬度在 [-90, 90] 之间的真实像元
    lat_full = h5s['geo']['Geolocation_Fields']['Latitude'][:]
    valid_rows, valid_cols = np.where((lat_full >= -90.0) & (lat_full <= 90.0))
    
    if len(valid_rows) == 0:
        print("⚠️ 警告：整轨数据为空（没有找到有效的经纬度）！")
        for f in h5s.values(): f.close()
        nc_out.close()
        return

    # 获取有效边界，并向外扩展 2 个像素作为安全冗余 (Padding)
    bbox_r_min = max(0, valid_rows.min() - 2)
    bbox_r_max = min(H, valid_rows.max() + 3)
    bbox_c_min = max(0, valid_cols.min() - 2)
    bbox_c_max = min(W, valid_cols.max() + 3)
    
    print(f"🎯 BBox 锁定！行: [{bbox_r_min}:{bbox_r_max}], 列: [{bbox_c_min}:{bbox_c_max}]")
    print(f"⚡ I/O 读取量将缩减至原来的 {((bbox_r_max-bbox_r_min)*(bbox_c_max-bbox_c_min))/(H*W)*100:.1f}%！")

    del lat_full 
    # =========================================================================
    # =========================================================================

    # ── 6. 逐行块读取 H5，提取特征，组装像元任务列表 ─────────────────────────
    all_pixel_tasks: List[dict] = []
    
    # 【修改】：循环范围不再是 0 到 H，而是 bbox_r_min 到 bbox_r_max
    n_blocks = (bbox_r_max - bbox_r_min + block_size - 1) // block_size

    try:
        for r_start in tqdm(range(bbox_r_min, bbox_r_max, block_size),
                            total=n_blocks, desc="📦 读取 & 提取特征"):
            r_end = min(bbox_r_max, r_start + block_size)
            
            # 【修改】：行切片是 rs，列切片是 cs！
            rs = slice(r_start, r_end)
            cs = slice(bbox_c_min, bbox_c_max)
            cur_h = r_end - r_start
            cur_w = bbox_c_max - bbox_c_min # 当前块的实际宽度

            # 6.1 经纬度 & 高程
            # 【修改】：所有读取都要加上 cs
            lat_blk  = h5s['geo']['Geolocation_Fields']['Latitude'][rs, cs]
            lon_blk  = h5s['geo']['Geolocation_Fields']['Longitude'][rs, cs]
            lat_buf[rs, cs] = lat_blk
            lon_buf[rs, cs] = lon_blk

            if 'Surface_Altitude' in h5s['geo']['Geolocation_Fields']:
                elev_blk = (h5s['geo']['Geolocation_Fields']
                            ['Surface_Altitude'][rs, cs].astype(np.float32) * 0.001)
            else:
                elev_blk = np.zeros((cur_h, cur_w), dtype=np.float32)

            # 6.2 掩膜（陆地 + 区域 + 云）
            land_sea   = h5s['geo']['Geolocation_Fields']['Sea_Land_Flags'][rs, cs]
            nadir_idx  = 7
            i565_nadir = h5s['565']['Data_Fields']['I565'][nadir_idx, rs, cs]
            i490_nadir = h5s['490']['Data_Fields']['I490P'][nadir_idx, rs, cs]
            mask_2d    = build_mask(lat_blk, lon_blk, land_sea,
                                    i565_nadir, i490_nadir,
                                    lat_min, lat_max, lon_min, lon_max)
            mask_flat  = mask_2d.flatten()
            valid_count = int(mask_flat.sum())
            if valid_count == 0:
                continue

            # 6.3 读取全角度辐射/偏振（展平像元维）
            def rb(h5k, path, cal=None, is_ang=False):
                # 【修改】：加入 cs 切片，并且 reshape 时宽度使用 cur_w
                d = h5s[h5k][path][:, rs, cs].astype(np.float32)
                d *= SCALE_ANG if is_ang else SCALE_REF
                if cal and cal in CAL_COEFFS:
                    d *= CAL_COEFFS[cal]
                return d.reshape(d.shape[0], cur_h * cur_w)

            sza = rb('geo', 'Data_Fields/Sol_Zen_Ang',   is_ang=True)
            vza = rb('geo', 'Data_Fields/View_Zen_Ang',  is_ang=True)
            saa = rb('geo', 'Data_Fields/Sol_Azim_Ang',  is_ang=True)
            vaa = rb('geo', 'Data_Fields/View_Azim_Ang', is_ang=True)
            if sza.shape[0] == 1:
                sza = np.tile(sza, (vza.shape[0], 1))
            phi  = np.abs(saa - vaa)

            I443   = rb('443', 'Data_Fields/I443')
            I490   = rb('490', 'Data_Fields/I490P',  cal='490')
            Q490   = rb('490', 'Data_Fields/Q490P',  cal='490')
            U490   = rb('490', 'Data_Fields/U490P',  cal='490')
            I565   = rb('565', 'Data_Fields/I565',   cal='565')
            I670   = rb('670', 'Data_Fields/I670P',  cal='670')
            Q670   = rb('670', 'Data_Fields/Q670P',  cal='670')
            U670   = rb('670', 'Data_Fields/U670P',  cal='670')
            I865   = rb('865', 'Data_Fields/I865P',  cal='865')
            Q865   = rb('865', 'Data_Fields/Q865P',  cal='865')
            U865   = rb('865', 'Data_Fields/U865P',  cal='865')

            DOLP490 = calc_dolp(I490, Q490, U490)
            DOLP670 = calc_dolp(I670, Q670, U670)
            DOLP865 = calc_dolp(I865, Q865, U865)

            # 6.4 有效像元坐标
            flat_indices  = np.flatnonzero(mask_flat)
            # 【修改】：宽度使用 cur_w，并且列坐标需要加上 bbox_c_min 还原到全局
            rows_in_block, cols_in_block = np.unravel_index(flat_indices, (cur_h, cur_w))
            global_rows   = r_start + rows_in_block
            global_cols   = bbox_c_min + cols_in_block
            global_flat   = global_rows * W + global_cols

            flat_lat  = lat_blk.flatten()[mask_flat]
            flat_lon  = lon_blk.flatten()[mask_flat]
            flat_elev = elev_blk.flatten()[mask_flat]

# 6.5 读取 Transformer 先验
            vc_prior = read_vc_prior_nc(vc_prior_nc, flat_lat, flat_lon)
            
            # 【修改】：因为 aod_prior_buf 和 ssa_prior_buf 是一维数组 (H*W)，
            # 所以必须使用一维的 global_flat 索引！
            aod_prior_buf[global_flat] = vc_prior['aod']
            ssa_prior_buf[global_flat] = vc_prior['ssa']

            # 6.6 批量 VC DNN 推理（整块一次 forward）
            vc_feats = np.zeros((valid_count, VC_FEAT_DIM), dtype=np.float32)
            for k, p in enumerate(flat_indices):
                vc_feats[k] = build_vc_feat_single(
                    flat_elev[k],
                    sza, vza, phi,
                    I443, I490, DOLP490, I565, I670, DOLP670, I865, DOLP865, p)

            X_sc = sx_vc.transform(vc_feats)
            with torch.no_grad():
                y_sc = vc_model(torch.from_numpy(X_sc).float()).numpy()
            vc_pred = np.maximum(sy_vc.inverse_transform(y_sc), 0.005)

            # 【修改】：同样使用一维的 global_flat 索引
            prior_vc_buf['vc_BB'][global_flat] = vc_pred[:, 0]
            prior_vc_buf['vc_Urban'][global_flat] = vc_pred[:, 1]
            prior_vc_buf['vc_Ocean'][global_flat] = vc_pred[:, 2]
            prior_vc_buf['vc_Dust'][global_flat] = vc_pred[:, 3]

            # 6.7 逐像元切片 data_obs
            for k, p in enumerate(flat_indices):
                data_obs = build_data_obs_single(
                    sza, vza, phi,
                    I443, I490, DOLP490, I565, I670, DOLP670, I865, DOLP865, p)
                if data_obs.shape[0] == 0:
                    continue

                t_aod = float(vc_prior['aod'][k])
                t_ssa = float(vc_prior['ssa'][k])
                all_pixel_tasks.append({
                    'pixel_index'  : k,
                    'flat_idx'     : int(global_flat[k]),   
                    'data_obs'     : data_obs,
                    'elev'         : float(flat_elev[k]),
                    'lon'          : float(flat_lon[k]),
                    'lat'          : float(flat_lat[k]),
                    'dynamic_prior': {
                        'vc_BB'    : float(vc_pred[k, 0]),
                        'vc_Urban' : float(vc_pred[k, 1]),
                        'vc_Ocean' : float(vc_pred[k, 2]),
                        'vc_Dust'  : float(vc_pred[k, 3]),
                        'truth_aod': t_aod if np.isfinite(t_aod) else -999.0,
                        'truth_ssa': t_ssa if np.isfinite(t_ssa) else -999.0,
                    },
                })

    finally:
        for f in h5s.values():
            f.close()

    print(f"\n✅ 特征提取完毕，共 {len(all_pixel_tasks)} 个有效像元。")

    if len(all_pixel_tasks) == 0:
        print("⚠️ 无有效像元，跳过反演。")
        nc_out.close()
        return

    # ── 7. 按 batch_size 切分成子任务批次 ────────────────────────────────────
    batches = []
    for i in range(0, len(all_pixel_tasks), batch_size):
        chunk = all_pixel_tasks[i: i + batch_size]
        batches.append({
            'batch_id'      : i // batch_size,
            'pixel_list'    : chunk,
            'prior_nc_path' : prior_nc_path,
            'config'        : config,
            'model_dict'    : model_dict,
            'features_scaler': features_scaler,
        })
    print(f"   → 划分为 {len(batches)} 个批次，每批最多 {batch_size} 像元")
    print(f"   → 并行进程数：{n_processes}\n")

    # ── 8. ProcessPoolExecutor 并行 OE 反演 ──────────────────────────────────
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        future_map = {
            executor.submit(process_pixel_batch, b): b['batch_id']
            for b in batches
        }
        with tqdm(total=len(batches), desc="🔄 OE 反演") as pbar:
            for future in as_completed(future_map):
                bid = future_map[future]
                try:
                    batch_results = future.result()
                    for r in batch_results:
                        fi = r['flat_idx']
                        flag_buf[fi] = r['quality_flag']
                        cost_buf[fi] = r['final_cost']
                        if r['optimized_state'] is not None:
                            for si, sv in enumerate(config.state_vector_list):
                                sv_buf[sv][fi] = r['optimized_state'][si]
                except Exception as e:
                    print(f"[批次 {bid}] 异常：{e}")
                finally:
                    pbar.update(1)

    # ── 9. 【新增】批量预测最终的宏观光学参数 (AOD, SSA 等) ───────────────────
    print("\n🔮 正在批量计算反演后的宏观光学参数 (AOD, SSA, AAOD等)...")
    
    # 提取有效反演结果的 VC
    opt_vc_BB = sv_buf['vc_BB']
    opt_vc_Urban = sv_buf['vc_Urban']
    opt_vc_Ocean = sv_buf['vc_Ocean']
    opt_vc_Dust = sv_buf['vc_Dust']

    # 找到所有成功反演的索引
    valid_mask = np.isfinite(opt_vc_BB) & np.isfinite(opt_vc_Urban) & np.isfinite(opt_vc_Ocean) & np.isfinite(opt_vc_Dust)
    valid_idx = np.flatnonzero(valid_mask)

    if len(valid_idx) > 0:
        # 组装输入矩阵 (N_valid, 4)
        vc_input = np.vstack([
            opt_vc_BB[valid_idx],
            opt_vc_Urban[valid_idx],
            opt_vc_Ocean[valid_idx],
            opt_vc_Dust[valid_idx]
        ]).T 

        # 调用已加载的 AOD 模型
        model_aerosol, scaler_x_aerosol, scaler_y_aerosol = model_dict['aerosol_prop']

        # 批量预测
        x_scaled = scaler_x_aerosol.transform(vc_input)
        x_tensor = torch.from_numpy(x_scaled).float().to('cpu')
        with torch.no_grad():
            y_pred_scaled = model_aerosol(x_tensor).numpy()
        y_pred = scaler_y_aerosol.inverse_transform(y_pred_scaled)

        # 写入 NC 变量
        macro_names = ["AOD", "SSA", "AAOD", "MR", "MI", "fineAOD", "coarseAOD"]
        for i_m, m_name in enumerate(macro_names):
            v = nc_out.createVariable(m_name, 'f4', ('line', 'pixel'), zlib=True, fill_value=np.nan)
            buf = np.full(H * W, np.nan, dtype=np.float32)
            buf[valid_idx] = y_pred[:, i_m]
            v[:, :] = buf.reshape(H, W)
    else:
        print("⚠️ 没有有效的反演像元，跳过宏观参数计算。")


    # ── 10. 写入 NC 文件 ──────────────────────────────────────────────────────
    print("💾 正在写入 NetCDF4 文件...")
    lat_var[:, :] = lat_buf
    lon_var[:, :] = lon_buf
    
    # 写入反演状态量
    for sv in config.state_vector_list:
        state_vars[sv][:, :] = sv_buf[sv].reshape(H, W)
    cost_var[:, :] = cost_buf.reshape(H, W)
    flag_var[:, :] = flag_buf.reshape(H, W)
    
    # 写入 Prior Group 变量
    for sv in ['vc_BB', 'vc_Urban', 'vc_Ocean', 'vc_Dust']:
        prior_vars[sv][:, :] = prior_vc_buf[sv].reshape(H, W)
    prior_vars['AOD_550nm'][:, :] = aod_prior_buf.reshape(H, W)
    prior_vars['SSA_550nm'][:, :] = ssa_prior_buf.reshape(H, W)
    
    nc_out.close()

    elapsed = (time.time() - t0) / 60
    print(f"\n✅ 任务全部完成！结果保存至：{output_nc_path}")
    print(f"⏱️  总耗时：{elapsed:.2f} 分钟")

# ==============================================================================
# 直接修改下方参数运行（无需命令行）
# ==============================================================================
if __name__ == '__main__':

    # ── 必填路径 ──────────────────────────────────────────────────────────────
    L1_DIR            = '/media/amers/Seagate Backup Plus Drive/GF5B_DPC_2025/GF5B_DPC_20251111_022228_L10001129978/'
    OUTPUT            = '/media/amers/SSD_part1/whx/ResNet_forDPC/GF5B_DPC_20251111_OE.nc'
    FORWARD_MODEL_DIR = '/media/amers/SSD_part1/whx/ResNet_code/forward/V1/resnet_param/'
    PRIOR_NC_DIR      = '/media/amers/WHX/NNOE_POLDER/POLDER_data/priori_data/'

    # ── VC 模型路径 ───────────────────────────────────────────────────────────
    VC_MODEL_PATH = '/media/amers/SSD_part1/whx/ResNet_forDPC/dynamic_prior/vc_model_DPC.pth'
    SCALER_X_VC   = '/media/amers/SSD_part1/whx/ResNet_forDPC/dynamic_prior/dnn_feature_scaler_vc_DPC.joblib'
    SCALER_Y_VC   = '/media/amers/SSD_part1/whx/ResNet_forDPC/dynamic_prior/dnn_target_scaler_vc_DPC.joblib'

    # ── Transformer 先验 NC（可选，None = 不使用 AOD/SSA 约束）──────────────
    VC_PRIOR_NC = '/media/amers/SSD_part1/DPC_project/Inference_GF5B_DPC_Transformer_20251111_finetune.nc'

    # ── 反演区域 ──────────────────────────────────────────────────────────────
    LON_MIN, LON_MAX = 72.0, 92.0
    LAT_MIN, LAT_MAX = 8.0, 33.0

    # ── 性能参数 ──────────────────────────────────────────────────────────────
    BLOCK_SIZE  = 512   # 主进程逐行读取块大小（行数）
    N_PROCESSES = 86    # 子进程数（建议 ≤ 物理核数）
    BATCH_SIZE  = 64    # 每批像元数
                        # 建议起始值：20~50；太大→单批耗时长；太小→调度开销大

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT)), exist_ok=True)

    run_retrieval(
        l1_dir            = L1_DIR,
        output_nc_path    = OUTPUT,
        forward_model_dir = FORWARD_MODEL_DIR,
        prior_nc_dir      = PRIOR_NC_DIR,
        vc_model_path     = VC_MODEL_PATH,
        scaler_x_vc_path  = SCALER_X_VC,
        scaler_y_vc_path  = SCALER_Y_VC,
        vc_prior_nc       = VC_PRIOR_NC,
        lon_min     = LON_MIN,
        lon_max     = LON_MAX,
        lat_min     = LAT_MIN,
        lat_max     = LAT_MAX,
        block_size  = BLOCK_SIZE,
        n_processes = N_PROCESSES,
        batch_size  = BATCH_SIZE,
    )