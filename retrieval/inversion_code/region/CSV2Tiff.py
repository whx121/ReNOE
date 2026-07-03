import os
import numpy as np
import pandas as pd
from osgeo import gdal, osr

# 读取 CSV 数据
csv_file = '/media/amers/WHX/NNOE_POLDER/retrieval/region/test_results_large_area/2007-05-28T05-16-58_multi_pixel_Henan.csv'
df = pd.read_csv(csv_file)

# 提取经纬度和目标变量
lons = df['lon'].values
lats = df['lat'].values
AOD = df['AOD'].values
SSA = df['SSA'].values
AAOD = df['AAOD'].values
# iso_565 = df['iso_0.565'].values
# iso_670 = df['iso_0.67'].values
# final_cost = df['final_cost'].values
# k1 = df['k1'].values
# k2 = df['k2'].values
# 定义输出网格参数
lon_min, lon_max = lons.min(), lons.max()
lat_min, lat_max = lats.min(), lats.max()
grid_res = 0.08  # 网格分辨率（度）

# 计算网格大小
grid_lon = np.arange(lon_min, lon_max + grid_res, grid_res)
grid_lat = np.arange(lat_max, lat_min - grid_res, -grid_res)
nrows, ncols = len(grid_lat), len(grid_lon)

# 初始化栅格数据，全为 nodata 值
nodata_value = -9999
AOD_grid = np.full((nrows, ncols), nodata_value, dtype=np.float32)
SSA_grid = np.full((nrows, ncols), nodata_value, dtype=np.float32)
AAOD_grid = np.full((nrows, ncols), nodata_value, dtype=np.float32)
final_cost_grid = np.full((nrows, ncols), nodata_value, dtype=np.float32)
iso_565_grid = np.full((nrows, ncols), nodata_value, dtype=np.float32)
iso_670_grid = np.full((nrows, ncols), nodata_value, dtype=np.float32)
k1_grid = np.full((nrows, ncols), nodata_value, dtype=np.float32)
k2_grid = np.full((nrows, ncols), nodata_value, dtype=np.float32)

# 计算每个点在栅格中的索引
lon_idx = ((lons - lon_min) / grid_res + 0.5).astype(int)
lat_idx = ((lat_max - lats) / grid_res + 0.5).astype(int)

# 仅在观测点处填充值
for i in range(len(df)):
    AOD_grid[lat_idx[i], lon_idx[i]] = AOD[i]
    SSA_grid[lat_idx[i], lon_idx[i]] = SSA[i]
    AAOD_grid[lat_idx[i], lon_idx[i]] = AAOD[i]
    # final_cost_grid[lat_idx[i], lon_idx[i]] = final_cost[i]
    # # iso_565_grid[lat_idx[i], lon_idx[i]] = iso_565[i]
    # iso_670_grid[lat_idx[i], lon_idx[i]] = iso_670[i]
    # k1_grid[lat_idx[i], lon_idx[i]] = k1[i]
    # k2_grid[lat_idx[i], lon_idx[i]] = k2[i]

# 定义仿射变换参数
geotransform = (
    lon_min - (grid_res / 2),  # 左上角X坐标
    grid_res,                  # 东西方向分辨率
    0,                         # 旋转参数
    lat_max + (grid_res / 2),  # 左上角Y坐标
    0,                         # 旋转参数
    -grid_res                  # 南北方向分辨率
)

# 定义目标坐标系 WGS84 (EPSG:4326)
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
wkt = srs.ExportToWkt()

# 写入 GeoTIFF 文件的函数
def write_tiff(filename, array, geotransform, wkt, nodata):
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(filename, ncols, nrows, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(wkt)
    band = dataset.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(nodata)
    band.FlushCache()
    dataset = None  # 关闭数据集
# 定义输出目录
output_dir = '/media/amers/WHX/NNOE_POLDER/retrieval/region/tiff/'
os.makedirs(output_dir, exist_ok=True)

# 保存 AOD、SSA、AAOD 为 GeoTIFF 文件
write_tiff(os.path.join(output_dir, '2007-05-28T05-03-58_henan_AOD.tif'), AOD_grid, geotransform, wkt, nodata_value)

write_tiff(os.path.join(output_dir, '2007-05-28T05-03-58_henan_SSA.tif'), SSA_grid, geotransform, wkt, nodata_value)
# write_tiff(os.path.join(output_dir, 'AAOD_20080930_3.tif'), AAOD_grid, geotransform, wkt, nodata_value)
# write_tiff(os.path.join(output_dir, 'final_cost_2008-10-12f.tif'), final_cost_grid, geotransform, wkt, nodata_value)
# write_tiff(os.path.join(output_dir, '2008-10-14T05-07-19T05-03-58with_priori_iso67022.tif'), iso_670_grid, geotransform, wkt, nodata_value)
# write_tiff(os.path.join(output_dir, '2008-10-14T05-03-58with_priori_k122.tif'), k1_grid, geotransform, wkt, nodata_value)
# write_tiff(os.path.join(output_dir, '2008-10-14T05-03-58with_priori_k222.tif'), k2_grid, geotransform, wkt, nodata_value)



print("TIFF 文件已生成 (无插值)：")

