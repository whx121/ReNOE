"""
矢量化POLDER数据提取模块
解决I/O瓶颈问题的优化版本
"""

import xarray as xr
import h5py
import math
import numpy as np
import os
import warnings
from typing import Tuple, List, Dict, Any
from tqdm import tqdm


def vectorized_extract_polder_data(file_path: str, file_name: str,
                                   coordinates: List[Tuple[int, int]]) -> Dict[str, Any]:
    """
    【高效版】矢量化提取POLDER数据，使用HDF5切片一次性读取数据区域。

    Args:
        file_path: 文件路径
        file_name: 文件名
        coordinates: [(lin1, col1), (lin2, col2), ...] 目标像元坐标列表

    Returns:
        包含所有有效像元数据的字典
    """
    full_file_path = os.path.join(file_path, file_name)
    print(f"开始高效读取文件: {file_name}, 总坐标数: {len(coordinates)}")

    if not coordinates:
        return {}

    # 1. 确定要读取的最小矩形边界 (bounding box)
    lins = np.array([c[0] for c in coordinates])
    cols = np.array([c[1] for c in coordinates])
    lin_min, lin_max = lins.min(), lins.max()
    col_min, col_max = cols.min(), cols.max()

    print(f"  - 确定数据切片范围: 行({lin_min}-{lin_max}), 列({col_min}-{col_max})")

    # 将原始坐标转换为集合，以便快速查找
    target_coords_set = set(coordinates)

    # 准备存储结果的列表
    all_data = {
        'data_obs_list': [], 'elev_list': [], 'lon_list': [], 'lat_list': []
    }

    with h5py.File(full_file_path, 'r') as f:
        # 2. 一次性读取所有需要的数据集的矩形切片
        print("  - 正在批量读取所有相关数据集...")
        data_slices = _read_all_dataset_slices(f, lin_min, lin_max, col_min, col_max)

        print("  - 数据读取完毕，开始在内存中处理...")
        # 3. 在内存中遍历切片，只处理我们需要的像元
        for r_slice_idx in tqdm(range(data_slices['lons'].shape[0]), desc="内存数据处理"):
            for c_slice_idx in range(data_slices['lons'].shape[1]):

                # 将切片内的相对坐标转换回绝对坐标
                abs_lin = lin_min + r_slice_idx
                abs_col = col_min + c_slice_idx

                # 只处理在目标列表中的像元
                if (abs_lin, abs_col) not in target_coords_set:
                    continue

                # 提取当前像元的所有数据（全部来自内存，非常快）
                pixel_data_slice = {key: val[r_slice_idx, c_slice_idx] for key, val in data_slices.items()}

                # 应用过滤条件
                cloud_val = cloud_grasp(pixel_data_slice.get('cloud_indicator', 255))
                if cloud_val != 1 or pixel_data_slice.get('land_sea_flag', 0) != 100:
                    continue

                # 计算观测数据
                data_obs = _calculate_observations_from_slice(pixel_data_slice)

                if data_obs is not None and data_obs.shape[0] > 0:
                    all_data['data_obs_list'].append(data_obs)
                    all_data['elev_list'].append(pixel_data_slice['surface_altitude'] / 1000.0)
                    all_data['lon_list'].append(pixel_data_slice['lons'])
                    all_data['lat_list'].append(pixel_data_slice['lats'])

    print(f"矢量化读取完成，有效像元数: {len(all_data['data_obs_list'])}")
    return all_data


def _read_all_dataset_slices(f: h5py.File, r_min, r_max, c_min, c_max) -> Dict[str, np.ndarray]:
    """一次性读取所有HDF5数据集的切片"""
    slices = {}
    r_slice = slice(r_min, r_max + 1)
    c_slice = slice(c_min, c_max + 1)

    datasets_to_read = [
        ('Geolocation_Fields', 'Longitude', 'lons'),
        ('Geolocation_Fields', 'Latitude', 'lats'),
        ('Geolocation_Fields', 'surface_altitude', 'surface_altitude'),
        ('Geolocation_Fields', 'land_sea_flag', 'land_sea_flag'),
        ('Data_Fields', 'cloud_indicator', 'cloud_indicator'),
        ('Data_Directional_Fields', 'thetas', 'thetas'),
        ('Data_Directional_Fields', 'thetav', 'thetav'),
        ('Data_Directional_Fields', 'phi', 'phi'),
        ('Data_Directional_Fields', 'I443NP', 'I443NP'), ('Data_Directional_Fields', 'I490P', 'I490P'),
        ('Data_Directional_Fields', 'Q490P', 'Q490P'), ('Data_Directional_Fields', 'U490P', 'U490P'),
        ('Data_Directional_Fields', 'I565NP', 'I565NP'), ('Data_Directional_Fields', 'I670P', 'I670P'),
        ('Data_Directional_Fields', 'Q670P', 'Q670P'), ('Data_Directional_Fields', 'U670P', 'U670P'),
        ('Data_Directional_Fields', 'I865P', 'I865P'), ('Data_Directional_Fields', 'Q865P', 'Q865P'),
        ('Data_Directional_Fields', 'U865P', 'U865P'), ('Data_Directional_Fields', 'I1020NP', 'I1020NP')
    ]
    for group, dset, key in datasets_to_read:
        try:
            slices[key] = f[group][dset][r_slice, c_slice]
        except KeyError:
            print(f"警告: 找不到数据集 {group}/{dset}")
            # 创建一个形状匹配的空数组或默认值数组
            shape = (r_max - r_min + 1, c_max - c_min + 1)
            if dset.endswith('P') or dset.endswith('NP'):  # 假设多角度数据
                slices[key] = np.full(shape + (16,), -32767, dtype=np.int16)
            else:
                slices[key] = np.zeros(shape, dtype=np.int16)

    return slices


def _calculate_observations_from_slice(pixel_data: Dict[str, Any]) -> np.ndarray:
    """从内存中的数据切片计算单个像元的观测数据"""
    sza = mean_tif(pixel_data['thetas'], 65534, 0.0015)
    vza = mean_tif(pixel_data['thetav'], 65534, 0.0015)
    phi_raw = mean_tif(pixel_data['phi'], 65534, 0.006)
    phi = np.abs((phi_raw + 180) % 360 - 180)

    I443 = mean_tif(pixel_data['I443NP'], -32767, 0.0001)
    I490 = mean_tif(pixel_data['I490P'], -32767, 0.0001)
    Q490 = mean_tif(pixel_data['Q490P'], -32767, 0.0001)
    U490 = mean_tif(pixel_data['U490P'], -32767, 0.0001)
    I565 = mean_tif(pixel_data['I565NP'], -32767, 0.0001)
    I670 = mean_tif(pixel_data['I670P'], -32767, 0.0001)
    Q670 = mean_tif(pixel_data['Q670P'], -32767, 0.0001)
    U670 = mean_tif(pixel_data['U670P'], -32767, 0.0001)
    I865 = mean_tif(pixel_data['I865P'], -32767, 0.0001)
    Q865 = mean_tif(pixel_data['Q865P'], -32767, 0.0001)
    U865 = mean_tif(pixel_data['U865P'], -32767, 0.0001)
    I1020 = mean_tif(pixel_data['I1020NP'], -32767, 0.0001)

    cos_sza = np.cos(np.deg2rad(sza))
    # 避免除以0
    cos_sza[cos_sza < 1e-6] = np.nan

    reflectance_443 = I443 / cos_sza
    reflectance_490 = I490 / cos_sza
    reflectance_565 = I565 / cos_sza
    reflectance_670 = I670 / cos_sza
    reflectance_865 = I865 / cos_sza
    reflectance_1020 = I1020 / cos_sza

    I490[I490 < 1e-6] = np.nan
    I670[I670 < 1e-6] = np.nan
    I865[I865 < 1e-6] = np.nan

    dolp_490 = np.sqrt(Q490 ** 2 + U490 ** 2) / I490
    dolp_670 = np.sqrt(Q670 ** 2 + U670 ** 2) / I670
    dolp_865 = np.sqrt(Q865 ** 2 + U865 ** 2) / I865

    return package_observation(
        sza, vza, phi, reflectance_443, reflectance_490, dolp_490,
        reflectance_565, reflectance_670, dolp_670,
        reflectance_865, dolp_865, reflectance_1020
    )


# ==================== 原始辅助函数 ====================

def cloud_grasp(cloud):
    """云标识处理"""
    if cloud == 50:
        cloud_ = 0
    if cloud == 100:
        cloud_ = 0
    if cloud == 0:
        cloud_ = 1
    if cloud == 255:
        cloud_ = 0
    return cloud_


def package_observation(sza, vza, phi,
                        reflectance_443,
                        reflectance_490, dolp_490,
                        reflectance_565,
                        reflectance_670, dolp_670,
                        reflectance_865, dolp_865,
                        reflectance_1020,
                        drop_nan=True):
    """
    将多个角度观测数据按顺序组合为二维数组，每行数据格式为：
      [sza, vza, phi, reflectance_443, reflectance_490, dolp_490,
       reflectance_565, reflectance_670, dolp_670, reflectance_865, dolp_865, reflectance_1020]
    返回：
      data_obs: 一个二维 numpy 数组，形状为 (N_valid, 12)，其中 N_valid 为不含 NaN 的观测数目。
    """
    # 将所有输入按列组合，得到形状 (N, 12)
    data_obs = np.column_stack([
        sza, vza, phi,
        reflectance_443,
        reflectance_490, dolp_490,
        reflectance_565,
        reflectance_670, dolp_670,
        reflectance_865, dolp_865,
        reflectance_1020
    ])

    if drop_nan:
        # 仅保留不包含 NaN 的行
        valid_idx = np.isfinite(data_obs).all(axis=1)
        data_obs = data_obs[valid_idx, :]

    return data_obs


def calculate_row_col(raw_lon, raw_lat):
    """定义函数，由经纬度计算对应的行列号"""
    cal_row = round(18 * (90 - raw_lat) + 0.5)
    Ni = round(3240 * math.sin(math.radians((cal_row - 0.5) / 18)))
    cal_col = round(3240.5 + Ni * raw_lon / 180)
    return int(cal_row), int(cal_col)


def extract_prior_data(lon_val, lat_val, file_path):
    """先验数据提取"""
    # 读取 NetCDF 文件
    ds = xr.open_dataset(file_path)

    def latlon_to_index(lat, lon, resolution=0.1):
        """
        将经纬度转换为网格索引：
          - 经度索引：从 -180° 开始，每个格网宽度为 resolution。
          - 纬度索引：从 -90° 开始，每个格网高度为 resolution。
        """
        x_index = int(round((lon + 180) / resolution))
        y_index = np.abs(int(round((lat - 70) / resolution)))
        return x_index, y_index

    # 计算格网索引
    x_idx, y_idx = latlon_to_index(lat_val, lon_val, resolution=0.1)

    # 提取数据
    k1 = ds["k1"].isel(x=x_idx, y=y_idx).values
    k2 = ds["k2"].isel(x=x_idx, y=y_idx).values
    BPDF = ds["BPDF"].isel(x=x_idx, y=y_idx).values
    ALH = ds["ALH"].isel(x=x_idx, y=y_idx).values
    iso_443 = ds["iso_443"].isel(x=x_idx, y=y_idx).values
    iso_490 = ds["iso_490"].isel(x=x_idx, y=y_idx).values
    iso_565 = ds["iso_565"].isel(x=x_idx, y=y_idx).values
    iso_670 = ds["iso_670"].isel(x=x_idx, y=y_idx).values
    iso_865 = ds["iso_865"].isel(x=x_idx, y=y_idx).values
    iso_1020 = ds["iso_1020"].isel(x=x_idx, y=y_idx).values

    k1 = k1 / iso_670
    k2 = k2 / iso_670

    # 处理 NaN 值
    iso_443 = 0.02 if np.isnan(iso_443) else iso_443
    iso_490 = 0.04 if np.isnan(iso_490) else iso_490
    iso_565 = 0.06 if np.isnan(iso_565) else iso_565
    iso_670 = 0.1 if np.isnan(iso_670) else iso_670
    iso_865 = 0.3 if np.isnan(iso_865) else iso_865
    iso_1020 = 0.6 if np.isnan(iso_1020) else iso_1020
    k1 = 0.6 if np.isnan(k1) else k1
    k2 = 0.4 if np.isnan(k2) else k2
    BPDF = 4 if np.isnan(BPDF) else BPDF
    ALH = 4500 if np.isnan(ALH) else ALH

    return iso_443, iso_490, iso_565, iso_670, iso_865, iso_1020, k1, k2, BPDF, ALH / 1000


# ==================== 保留原始函数以保持兼容性 ====================

def mean_tif(a, nan_value, scale, flag=False):
    """原始的数据处理函数"""
    # 类型转换为float
    a = a.astype('float')
    # 将测量值中是无效值的赋为nan
    a[a == nan_value] = np.nan
    a[a == np.abs(nan_value)] = np.nan  # 实际数据有多个无效值 见POLDER文件
    a[a == nan_value + 1] = np.nan

    # 过滤特定警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # 计算真实值（像元值*scale_factor）
        mean_value = a * scale

    # 找到数组中非nan值位置
    index_ = np.argwhere(~np.isnan(mean_value))
    if flag:
        return mean_value, index_
    else:
        return mean_value


def extract_cloud(file_path, file_name, lin, col):
    """提取云信息"""
    f = h5py.File(os.path.join(file_path, file_name), 'r')
    cloud = cloud_grasp(f['Geolocation_Fields']['cloud_indicator'][lin, col])
    f.close()
    return cloud


def extract_polder_h5_(file_path, file_name, lin, col):
    """原始的单像元提取函数，保持向后兼容"""
    f = h5py.File(os.path.join(file_path, file_name), 'r')

    # 读取各参数所有行列号的值
    cloud = extract_cloud(file_path, file_name, lin, col)
    elev = f['Geolocation_Fields']['surface_altitude'][lin, col] / 1000
    land_sea = f['Geolocation_Fields']['land_sea_flag'][lin, col]

    sza, index_sza = mean_tif(f['Data_Directional_Fields']['thetas'][lin, col], 65534, 0.0015, flag=True)
    vza, index_vza = mean_tif(f['Data_Directional_Fields']['thetav'][lin, col], 65534, 0.0015, flag=True)
    phi, index_phi = mean_tif(f['Data_Directional_Fields']['phi'][lin, col], 65534, 0.006, flag=True)
    phi = np.abs((phi + 180) % 360 - 180)  # 映射到 [0, 180]

    I443, index_I443 = mean_tif(f['Data_Directional_Fields']['I443NP'][lin, col], -32767, 0.0001, flag=True)
    I490, index_I490 = mean_tif(f['Data_Directional_Fields']['I490P'][lin, col], -32767, 0.0001, flag=True)
    Q490, index_Q490 = mean_tif(f['Data_Directional_Fields']['Q490P'][lin, col], -32767, 0.0001, flag=True)
    U490, index_U490 = mean_tif(f['Data_Directional_Fields']['U490P'][lin, col], -32767, 0.0001, flag=True)
    I565, index_I565 = mean_tif(f['Data_Directional_Fields']['I565NP'][lin, col], -32767, 0.0001, flag=True)
    I670, index_I670 = mean_tif(f['Data_Directional_Fields']['I670P'][lin, col], -32767, 0.0001, flag=True)
    Q670, index_Q670 = mean_tif(f['Data_Directional_Fields']['Q670P'][lin, col], -32767, 0.0001, flag=True)
    U670, index_U670 = mean_tif(f['Data_Directional_Fields']['U670P'][lin, col], -32767, 0.0001, flag=True)
    I865, index_I865 = mean_tif(f['Data_Directional_Fields']['I865P'][lin, col], -32767, 0.0001, flag=True)
    Q865, index_Q865 = mean_tif(f['Data_Directional_Fields']['Q865P'][lin, col], -32767, 0.0001, flag=True)
    U865, index_U865 = mean_tif(f['Data_Directional_Fields']['U865P'][lin, col], -32767, 0.0001, flag=True)
    I1020, index_I1020 = mean_tif(f['Data_Directional_Fields']['I1020NP'][lin, col], -32767, 0.0001, flag=True)

    # 计算refectance 与 DOLP
    reflectance_443 = I443 / np.cos(np.deg2rad(sza))
    reflectance_490 = I490 / np.cos(np.deg2rad(sza))
    reflectance_565 = I565 / np.cos(np.deg2rad(sza))
    reflectance_670 = I670 / np.cos(np.deg2rad(sza))
    reflectance_865 = I865 / np.cos(np.deg2rad(sza))
    reflectance_1020 = I1020 / np.cos(np.deg2rad(sza))

    dolp_490 = np.sqrt(Q490 ** 2 + U490 ** 2) / I490
    dolp_670 = np.sqrt(Q670 ** 2 + U670 ** 2) / I670
    dolp_865 = np.sqrt(Q865 ** 2 + U865 ** 2) / I865
    f.close()

    #   返回迭代优化所需要的数据类型 shape(N_valid_angle, 12)
    data_obs = package_observation(sza, vza, phi, reflectance_443, reflectance_490, dolp_490,
                                   reflectance_565, reflectance_670, dolp_670, reflectance_865, dolp_865,
                                   reflectance_1020)
    return data_obs, cloud, elev, land_sea


if __name__ == '__main__':
    # 测试代码
    file_path = '/media/amers/2E42853942850735/whx/Doctor/NNOE_polder/retrieval/data'
    filename = 'POLDER3_L1B-BG1-089018M_2008-10-14T05-03-58_V1-01.h5'

    # 测试单像元提取
    lin, col = calculate_row_col(117.3, 38.9)
    data_obs, cloud, elev, land_sea = extract_polder_h5_(file_path, filename, lin, col)
    print(f"单像元测试: shape={data_obs.shape}, cloud={cloud}, elev={elev}")

    # 测试矢量化提取
    coordinates = [(lin, col), (lin + 1, col), (lin, col + 1)]
    vectorized_data = vectorized_extract_polder_data(file_path, filename, coordinates)
    print(f"矢量化测试: 有效像元数={len(vectorized_data['data_obs_list'])}")