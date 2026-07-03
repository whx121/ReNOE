import time
import xarray as xr
import h5py
import math
import numpy as np
import  os
import  warnings
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

# 定义函数，由经纬度计算对应的行列号
def calculate_row_col(raw_lon, raw_lat):
    cal_row = round(18 * (90 - raw_lat) + 0.5)
    Ni = round(3240 * math.sin(math.radians((cal_row - 0.5) / 18)))
    cal_col = round(3240.5 + Ni * raw_lon / 180)
    return int(cal_row), int(cal_col)

def cloud_grasp(cloud):
    if cloud == 50:
        cloud_ = 0
    if cloud == 100:
        cloud_ = 0
    if cloud == 0:
        cloud_ = 1
    if cloud == 255:
        cloud_ = 0
    return cloud_

def extract_cloud(file_path, file_name, lin, col):
    f = h5py.File(os.path.join(file_path, file_name), 'r')
    cloud = cloud_grasp(f['Data_Fields']['cloud_indicator'][lin, col])
    return cloud

def mean_tif(a, nan_value, scale, flag=False):
    # 类型转换为float
    a = a.astype('float')
    # 将测量值中是无效值的赋为nan
    a[a == nan_value] = np.nan
    a[a == np.abs(nan_value)] = np.nan  # 实际数据有多个无效值 见POLDER文件
    a[a == nan_value + 1] = np.nan
    # a[a == 0] = np.nan
    # 过滤特定警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # 计算真实值（像元值*scale_factor）
        # mean_value = np.nanmean(a, axis=(0, 1)) * scale
        mean_value = a * scale
    # 找到数组中非nan值位置
    index_ = np.argwhere(~np.isnan(mean_value))
    if flag:
        return mean_value, index_
    else:
        return mean_value

def extract_polder_h5_(file_path, file_name, lin, col):
    f = h5py.File(os.path.join(file_path, file_name), 'r')
    date_t = file_name.split('_')[2]
    date_year_month_day = date_t.split('T')[0]  # 2008-06-14
    date_h_m_s = date_t.split('T')[1]  # 14-27-43
    date_hour_min_sec = date_h_m_s.replace('-', ':')  # 14:27:13
    date_time = date_year_month_day + 'T' + date_hour_min_sec + 'Z'  # 2008-06-14T14:27:13Z

    # 读取各参数所有行列号的值
    cloud = extract_cloud(file_path, file_name, lin, col)
    # lon = f['Geolocation_Fields']['Longitude'][lin, col]
    # lat = f['Geolocation_Fields']['Latitude'][lin, col]
    elev = f['Geolocation_Fields']['surface_altitude'][lin, col]/1000
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
    data_obs = package_observation(sza, vza, phi, reflectance_443,reflectance_490, dolp_490,
                                   reflectance_565, reflectance_670, dolp_670, reflectance_865, dolp_865, reflectance_1020 )
    return data_obs, cloud, elev, land_sea

# def extract_polder_h5_(file_path, file_name, lin, col):
#     f = h5py.File(os.path.join(file_path, file_name), 'r')
#     date_t = file_name.split('_')[2]
#     date_year_month_day = date_t.split('T')[0]  # 2008-06-14
#     date_h_m_s = date_t.split('T')[1]  # 14-27-43
#     date_hour_min_sec = date_h_m_s.replace('-', ':')  # 14:27:13
#     date_time = date_year_month_day + 'T' + date_hour_min_sec + 'Z'  # 2008-06-14T14:27:13Z
#
#     # 读取各参数所有行列号的值
#     cloud = extract_cloud(file_path, file_name, lin, col)
#     # lon = f['Geolocation_Fields']['Longitude'][lin, col]
#     # lat = f['Geolocation_Fields']['Latitude'][lin, col]
#     elev = f['Geolocation_Fields']['surface_altitude'][lin, col]/1000
#     land_sea = f['Geolocation_Fields']['land_sea_flag'][lin, col]
#
#     sza, index_sza = mean_tif(f['Data_Directional_Fields']['thetas'][lin, col][6:10], 65534, 0.0015, flag=True)
#     vza, index_vza = mean_tif(f['Data_Directional_Fields']['thetav'][lin, col][6:10], 65534, 0.0015, flag=True)
#     phi, index_phi = mean_tif(f['Data_Directional_Fields']['phi'][lin, col][6:10], 65534, 0.006, flag=True)
#
#     I443, index_I443 = mean_tif(f['Data_Directional_Fields']['I443NP'][lin, col][6:10], -32767, 0.0001, flag=True)
#     I490, index_I490 = mean_tif(f['Data_Directional_Fields']['I490P'][lin, col][6:10], -32767, 0.0001, flag=True)
#     Q490, index_Q490 = mean_tif(f['Data_Directional_Fields']['Q490P'][lin, col][6:10], -32767, 0.0001, flag=True)
#     U490, index_U490 = mean_tif(f['Data_Directional_Fields']['U490P'][lin, col][6:10], -32767, 0.0001, flag=True)
#     I565, index_I565 = mean_tif(f['Data_Directional_Fields']['I565NP'][lin, col][6:10], -32767, 0.0001, flag=True)
#     I670, index_I670 = mean_tif(f['Data_Directional_Fields']['I670P'][lin, col][6:10], -32767, 0.0001, flag=True)
#     Q670, index_Q670 = mean_tif(f['Data_Directional_Fields']['Q670P'][lin, col][6:10], -32767, 0.0001, flag=True)
#     U670, index_U670 = mean_tif(f['Data_Directional_Fields']['U670P'][lin, col][6:10], -32767, 0.0001, flag=True)
#     I865, index_I865 = mean_tif(f['Data_Directional_Fields']['I865P'][lin, col][6:10], -32767, 0.0001, flag=True)
#     Q865, index_Q865 = mean_tif(f['Data_Directional_Fields']['Q865P'][lin, col][6:10], -32767, 0.0001, flag=True)
#     U865, index_U865 = mean_tif(f['Data_Directional_Fields']['U865P'][lin, col][6:10], -32767, 0.0001, flag=True)
#     I1020, index_I1020 = mean_tif(f['Data_Directional_Fields']['I1020NP'][lin, col][6:10], -32767, 0.0001, flag=True)
#
#     # 计算refectance 与 DOLP
#     reflectance_443 = I443 / np.cos(np.deg2rad(vza))
#     reflectance_490 = I490 / np.cos(np.deg2rad(vza))
#     reflectance_565 = I565 / np.cos(np.deg2rad(vza))
#     reflectance_670 = I670 / np.cos(np.deg2rad(vza))
#     reflectance_865 = I865 / np.cos(np.deg2rad(vza))
#     reflectance_1020 = I1020 / np.cos(np.deg2rad(vza))
#
#     dolp_490 = np.sqrt(Q490 ** 2 + U490 ** 2) / I490
#     dolp_670 = np.sqrt(Q670 ** 2 + U670 ** 2) / I670
#     dolp_865 = np.sqrt(Q865 ** 2 + U865 ** 2) / I865
#     f.close()
#
#     #   返回迭代优化所需要的数据类型 shape(N_valid_angle, 12)
#     data_obs = package_observation(sza, vza, phi, reflectance_443,reflectance_490, dolp_490,
#                                    reflectance_565, reflectance_670, dolp_670, reflectance_865, dolp_865, reflectance_1020,)
#     return data_obs, cloud, elev, land_sea



# 先验数据提取
# 数据集的维度名称为 "x" 和 "y"（格网编号），格网分辨率为0.1°
def extract_prior_data(lon_val,lat_val,file_path):
    # 读取 NetCDF 文件
    #file_path = "/media/amers/WHX/POLDER_data/GRASP_priori_data_10.nc"
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
    # 根据格网索引提取数据
    # 注意：如果数据中维度顺序不是 (y, x)，需要相应调整
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
    k1 = k1/iso_670
    k2 = k2/iso_670
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
    return iso_443, iso_490, iso_565, iso_670, iso_865, iso_1020,k1, k2, BPDF, ALH/1000
    # 注：这里的k1 k2 是normalised 的 vol与geo  目前不确定是不是/iso的  ALH是mean height

if __name__ == '__main__':
    file_path = '/media/amers/2E42853942850735/whx/Doctor/NNOE_polder/retrieval/data'
    filename = 'POLDER3_L1B-BG1-089018M_2008-10-14T05-03-58_V1-01.h5'
    lin,col = calculate_row_col(117.3,38.9)
    extract_polder_h5_(file_path, filename, lin,col)