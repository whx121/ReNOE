import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
from osgeo import gdal, osr

# --- START: 解决PROJ库路径问题的代码 (建议保留) ---
try:
    proj_lib_path = os.path.join(sys.prefix, 'share', 'proj')
    if os.path.exists(proj_lib_path):
        os.environ['PROJ_LIB'] = proj_lib_path
except Exception:
    print("警告: 无法自动设置PROJ_LIB环境变量。")
# --- END ---


def grid_and_save(lons: np.ndarray, lats: np.ndarray, values: np.ndarray,
                  lon_bins: np.ndarray, lat_bins: np.ndarray,
                  output_filename: str):
    """
    将点数据进行网格化平均，并保存为GeoTIFF文件。

    参数:
    lons (np.ndarray): 经度数组.
    lats (np.ndarray): 纬度数组.
    values (np.ndarray): 要进行网格化的数值数组 (如AOD, SSA).
    lon_bins (np.ndarray): 经度网格的边界.
    lat_bins (np.ndarray): 纬度网格的边界.
    output_filename (str): 输出的TIFF文件名 (包含完整路径).
    """
    print(f"正在处理: {os.path.basename(output_filename)}...")

    # 1. 使用 binned_statistic_2d 进行网格化和平均
    statistic, _, _, _ = binned_statistic_2d(
        lons, lats, values,
        statistic='mean',
        bins=[lon_bins, lat_bins]
    )

    # 2. 调整数组方向以匹配地理坐标系
    gridded_data = np.flipud(statistic.T)

    # 3. 替换nodata值
    nodata_value = -9999
    gridded_data = np.nan_to_num(gridded_data, nan=nodata_value)

    # 4. 定义地理参考信息
    grid_res = lon_bins[1] - lon_bins[0]
    geotransform = (
        lon_bins[0],      # 左上角X坐标
        grid_res,         # 东西方向分辨率
        0,
        lat_bins[-1],     # 左上角Y坐标
        0,
        -grid_res         # 南北方向分辨率
    )

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    wkt = srs.ExportToWkt()

    # 5. 写入 GeoTIFF 文件
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        output_filename,
        gridded_data.shape[1],  # ncols
        gridded_data.shape[0],  # nrows
        1,
        gdal.GDT_Float32
    )
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(wkt)
    band = dataset.GetRasterBand(1)
    band.WriteArray(gridded_data)
    band.SetNoDataValue(nodata_value)
    band.FlushCache()
    dataset = None
    print(f"文件已成功保存: {output_filename}")


# ==============================================================================
# 主程序
# ==============================================================================

if __name__ == "__main__":
    # 读取 CSV 数据
    csv_path = '/media/amers/WHX/NNOE_POLDER/retrieval/region/test_results_large_area/'
    csv_name_list = [#'2006-03-18T05-32-07_single.csv',
                    #  '2006-03-18T05-32-07_single_br.csv',
                     '2007-05-28T05-16-58_multi_pixel_Henan.csv']
        #'2006-05-08T09-26-44_single_indian.csv',
                     # '2006-05-08T11-05-36_single_indian.csv',
                     # '2006-05-08T12-44-30_single_indian.csv',
                     # '2006-05-08T14-23-24_single_indian.csv',
                     # '2006-05-08T06-08-57_single_indian.csv',
                     # '2006-05-08T07-47-50_single_indian.csv',]

    for csv_name in csv_name_list:


        polder_time = csv_name.split('T')[0].replace('-', '')

        df = pd.read_csv(os.path.join(csv_path, csv_name))
        df = df[df['final_cost']<10]
        df = df[df['AOD']>-0.05]
        # 定义输出目录
        output_dir = '/media/amers/WHX/NNOE_POLDER/retrieval/region/论文case/ResNOE/20070528/'
        os.makedirs(output_dir, exist_ok=True)

        # --- 数据准备 ---
        # 定义要处理的变量列表和对应的输出文件名
        variables_to_process = {
            "AOD": f"{polder_time}_AOD.tif",
            "SSA": f"{polder_time}_SSA.tif",
            "fineAOD": f"{polder_time}_fineAOD.tif",
            "coarseAOD": f"{polder_time}_coarseAOD.tif",


        }

        # 预处理：一次性移除所有目标变量中含有NaN的行
        valid_columns = ['lon', 'lat'] + list(variables_to_process.keys())
        df_clean = df.dropna(subset=valid_columns)

        if df_clean.empty:
            print("错误: 清理数据后没有任何有效行，请检查CSV文件内容。")
        else:
            # 提取经纬度
            lons = df_clean['lon'].values
            lats = df_clean['lat'].values

            # --- 网格定义 (所有变量共用) ---
            lon_min, lon_max = lons.min(), lons.max()
            lat_min, lat_max = lats.min(), lats.max()
            grid_res = 0.1  # 网格分辨率（度）

            lon_bins = np.arange(lon_min, lon_max + grid_res, grid_res)
            lat_bins = np.arange(lat_min, lat_max + grid_res, grid_res)

            # --- 循环处理每个变量 ---
            for var_name, file_name in variables_to_process.items():
                # 提取当前要处理的数值
                values = df_clean[var_name].values
                output_filename = os.path.join(output_dir, file_name)

                # 调用函数进行栅格化和保存
                grid_and_save(lons, lats, values, lon_bins, lat_bins, output_filename)

        print("\n所有处理完成！")