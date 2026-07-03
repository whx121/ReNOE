#!/usr/bin/env python3
"""
大范围区域反演主函数 - 基于CPU绑定的分块处理策略
学习站点并行处理的成功经验，避免I/O瓶颈和内存问题
"""

import os
import numpy as np
import pandas as pd
from time import time
import warnings
import logging
import joblib
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import json
import torch
import argparse
import sys
import multiprocessing as mp
import subprocess
import tempfile
import shutil

# 设置环境
warnings.filterwarnings('ignore', category=UserWarning)
device = torch.device('cpu')

# 导入自定义模块（保持与原代码相同的导入）
from retrieval.moudle.ResNet_RTModel_pytorch import DNNModel
import retrieval.moudle.moudle_OE as OEfunc
from retrieval.moudle.predict_forAOD import predict_AOD_AAOD_SSA as predict_AOD

# 导入所有需要的类和函数（与原代码相同）
from retrieval.inversion_code.aeronet.aeronet_inversion_single_pixel_pytorch_batchMod  import (
    RetrievalConfig, ImprovedOptimizer, compute_weights_improved,
    process_multi_angle_data_improved, update_config_with_prior,
    quality_check, process_single_pixel_improved,
    declare_multi_model, update_state_in_data, predict_multi_wavelength,
    current_total_cost_function_multi, print_retrieval_settings
)


class RegionalBlockProcessor:
    """大范围区域分块处理器"""

    def __init__(self, config: RetrievalConfig, base_dir: str, prior_path: str):
        self.config = config
        self.base_dir = base_dir
        self.prior_path = prior_path
        self.model_dict = None
        self.features_scaler = None

    def load_models(self):
        """加载模型（在每个进程中独立加载）"""
        if self.model_dict is None:
            self.model_dict = declare_multi_model(self.config.wl_list, self.base_dir, device='cpu')
            self.features_scaler = joblib.load(os.path.join(self.base_dir, 'scaler_features.pkl'))

    def process_coordinate_block(self, block_info: Dict) -> List[Dict]:
        """处理单个坐标块"""
        block_id = block_info['block_id']
        coordinates = block_info['coordinates']
        file_path = block_info['file_path']
        file_name = block_info['file_name']
        polder_time = block_info['polder_time']

        print(f"开始处理块 {block_id}，坐标数: {len(coordinates)}")

        # 确保模型已加载
        self.load_models()

        # 收集有效像元数据
        data_obs_list, elev_list, lon_list, lat_list = [], [], [], []

        for lon, lat in coordinates:
            try:
                lin, col = OEfunc.calculate_row_col(lon, lat)
                data_obs, cloud, elev, land_sea = OEfunc.extract_polder_h5_(
                    file_path, file_name, lin, col
                )

                # 检查数据有效性（无云陆地）
                if cloud == 1 and land_sea == 100:
                    data_obs_list.append(data_obs)
                    elev_list.append(elev)
                    lon_list.append(lon)
                    lat_list.append(lat)

            except Exception as e:
                continue

        print(f"块 {block_id} 有效像元数: {len(data_obs_list)}")

        if not data_obs_list:
            return []

        # 执行反演（使用单线程，避免嵌套并行）
        results = []
        for i in range(len(data_obs_list)):
            try:
                result = process_single_pixel_improved(
                    i, data_obs_list[i], elev_list[i], lon_list[i], lat_list[i],
                    self.prior_path, self.config, self.model_dict,
                    self.features_scaler, priori_flag=True
                )

                # 添加时间和位置信息
                if 'optimized_state' in result:
                    result['time'] = polder_time
                    result['lon'] = lon_list[i]
                    result['lat'] = lat_list[i]
                else:
                    result['lon'] = lon_list[i]
                    result['lat'] = lat_list[i]

                results.append(result)

            except Exception as e:
                results.append({
                    'pixel_index': i,
                    'lon': lon_list[i],
                    'lat': lat_list[i],
                    'error': str(e)
                })

        print(f"块 {block_id} 处理完成，结果数: {len(results)}")
        return results


def create_coordinate_blocks(lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                             block_size_deg: float = 1.0) -> List[List[Tuple[float, float]]]:
    """将大区域分割成小块坐标"""
    # 生成所有坐标点
    lat_range = np.arange(lat_max * 100, lat_min * 100, -5) * 0.01
    lon_range = np.arange(lon_min * 100, lon_max * 100, 5) * 0.01

    all_coordinates = [(lon, lat) for lat in lat_range for lon in lon_range]

    # 按地理位置分块
    blocks = []
    lat_blocks = int(np.ceil((lat_max - lat_min) / block_size_deg))
    lon_blocks = int(np.ceil((lon_max - lon_min) / block_size_deg))

    for i in range(lat_blocks):
        for j in range(lon_blocks):
            block_lat_min = lat_min + i * block_size_deg
            block_lat_max = min(block_lat_min + block_size_deg, lat_max)
            block_lon_min = lon_min + j * block_size_deg
            block_lon_max = min(block_lon_min + block_size_deg, lon_max)

            # 筛选属于当前块的坐标
            block_coords = [
                (lon, lat) for lon, lat in all_coordinates
                if block_lon_min <= lon < block_lon_max and block_lat_min <= lat < block_lat_max
            ]

            if block_coords:
                blocks.append(block_coords)

    return blocks


def run_block_with_cpu_binding(block_info: Dict, cpu_cores: str, output_dir: str) -> str:
    """使用CPU绑定运行单个块的处理"""
    import pickle

    # 创建临时文件传递数据
    temp_dir = tempfile.mkdtemp()
    block_file = os.path.join(temp_dir, f"block_{block_info['block_id']}.pkl")
    result_file = os.path.join(temp_dir, f"result_{block_info['block_id']}.pkl")

    # 保存块信息
    with open(block_file, 'wb') as f:
        pickle.dump(block_info, f)

    # 构建python脚本内容
    script_content = f'''
import sys
import os
import pickle
import numpy as np
sys.path.insert(0, "/media/amers/ssd_1t/whx/ResNet_code/")

from {__name__} import RegionalBlockProcessor, RetrievalConfig

# 设置环境变量
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 加载数据
with open("{block_file}", "rb") as f:
    block_info = pickle.load(f)

# 创建处理器
config = RetrievalConfig()
processor = RegionalBlockProcessor(
    config, 
    "{block_info['base_dir']}", 
    "{block_info['prior_path']}"
)

# 处理块
try:
    results = processor.process_coordinate_block(block_info)
    with open("{result_file}", "wb") as f:
        pickle.dump(results, f)
    print(f"块 {{block_info['block_id']}} 处理成功")
except Exception as e:
    print(f"块 {{block_info['block_id']}} 处理失败: {{e}}")
    with open("{result_file}", "wb") as f:
        pickle.dump([], f)
'''

    script_file = os.path.join(temp_dir, f"process_block_{block_info['block_id']}.py")
    with open(script_file, 'w') as f:
        f.write(script_content)

    # 使用taskset绑定CPU核心运行
    cmd = [
        'taskset', '-c', cpu_cores,
        'python', script_file
    ]

    log_file = os.path.join(output_dir, f"block_{block_info['block_id']}.log")

    try:
        with open(log_file, 'w') as log:
            process = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3600,  # 1小时超时
                cwd="/media/amers/ssd_1t/whx/ResNet_code/"
            )

        # 读取结果
        if os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                results = pickle.load(f)
        else:
            results = []

        return results

    except subprocess.TimeoutExpired:
        print(f"块 {block_info['block_id']} 处理超时")
        return []
    except Exception as e:
        print(f"块 {block_info['block_id']} 执行失败: {e}")
        return []
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir, ignore_errors=True)


def allocate_cpu_cores(total_cores: int, n_processes: int) -> List[str]:
    """分配CPU核心给各个进程"""
    cores_per_process = max(1, total_cores // n_processes)
    core_allocations = []

    for i in range(n_processes):
        start_core = i * cores_per_process
        end_core = min(start_core + cores_per_process - 1, total_cores - 1)
        if start_core <= end_core:
            core_allocations.append(f"{start_core}-{end_core}")
        else:
            core_allocations.append(str(start_core))

    return core_allocations


if __name__ == "__main__":
    """
    大范围区域反演主函数
    使用CPU绑定和分块处理策略，避免I/O瓶颈
    """
    T_begin = time()

    # 解析命令行参数（如果需要）
    parser = argparse.ArgumentParser(description='POLDER大范围区域反演')
    parser.add_argument('--n_processes', type=int, default=24, help='并行进程数')
    parser.add_argument('--block_size', type=float, default=2.0, help='块大小（度）')
    parser.add_argument('--total_cores', type=int, default=96, help='总CPU核心数')
    args = parser.parse_args()

    print(f"系统配置: {mp.cpu_count()} CPU核心")
    print(f"使用配置: {args.n_processes} 进程, 块大小 {args.block_size}°")

    # 创建配置对象
    config = RetrievalConfig()
    print_retrieval_settings(config)

    # 路径配置
    file_path = r"/media/amers/ssd_1t/whx/ResNet_code/retrieval/test_data/"
    file_name_list = [
        'POLDER3_L1B-BG1-088222M_2008-10-12T05-16-33_V1-01.h5'
    ]
    output_path = r'/media/amers/WHX/NNOE_POLDER/retrieval/region/test_results/'
    os.makedirs(output_path, exist_ok=True)

    # 模型和先验路径
    base_dir = '/media/amers/ssd_1t/whx/ResNet_code/forward/resnet_param/'
    prior_path = '/media/amers/WHX/NNOE_POLDER/POLDER_data/priori_data/GRASP_priori_data_10.nc'

    # CPU核心分配
    core_allocations = allocate_cpu_cores(args.total_cores, args.n_processes)
    print(f"CPU核心分配: {core_allocations}")

    # 处理每个文件
    for file_name in file_name_list:
        print(f'\n开始处理文件：{file_name}')

        # 提取时间信息
        polder_time = file_name.split('_')[2].split('T')[0]
        output_name = f"{file_name.split('_')[2]}_regional_cpu_bound.csv"

        # 定义研究区域（京津冀）
        lat_max, lat_min = 42, 35
        lon_min, lon_max = 114, 121

        print(f"处理区域: 经度{lon_min}°-{lon_max}°, 纬度{lat_min}°-{lat_max}°")

        # 创建坐标块
        coordinate_blocks = create_coordinate_blocks(
            lat_min, lat_max, lon_min, lon_max, args.block_size
        )

        total_coords = sum(len(block) for block in coordinate_blocks)
        print(f"总坐标数: {total_coords}, 分为 {len(coordinate_blocks)} 个块")

        # 准备块信息
        block_infos = []
        for i, coords in enumerate(coordinate_blocks):
            block_info = {
                'block_id': i,
                'coordinates': coords,
                'file_path': file_path,
                'file_name': file_name,
                'polder_time': polder_time,
                'base_dir': base_dir,
                'prior_path': prior_path
            }
            block_infos.append(block_info)

        # 分批处理块（每批使用所有可用进程）
        all_results = []
        batch_size = args.n_processes

        for batch_start in range(0, len(block_infos), batch_size):
            batch_end = min(batch_start + batch_size, len(block_infos))
            current_batch = block_infos[batch_start:batch_end]

            print(f"\n处理批次 {batch_start // batch_size + 1}/{(len(block_infos) + batch_size - 1) // batch_size}")
            print(f"块 {batch_start} 到 {batch_end - 1}")

            # 并行处理当前批次
            batch_results = []
            processes = []

            for i, block_info in enumerate(current_batch):
                cpu_cores = core_allocations[i % len(core_allocations)]

                # 使用subprocess运行独立进程
                proc_results = run_block_with_cpu_binding(
                    block_info, cpu_cores, output_path
                )
                batch_results.extend(proc_results)

            print(f"批次完成，获得结果数: {len(batch_results)}")
            all_results.extend(batch_results)

        # 保存最终结果
        if all_results:
            print(f"\n处理完成，总结果数: {len(all_results)}")

            # 转换为DataFrame并保存
            processed_results = []
            for result in all_results:
                if 'error' in result:
                    result_data = {
                        'time': polder_time,
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
                        'time': result.get('time', polder_time),
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

            result_df = pd.DataFrame(processed_results)
            result_file = os.path.join(output_path, output_name)
            result_df.to_csv(result_file, index=False)

            # 统计信息
            successful_pixels = sum(1 for r in processed_results if r['error'] == '')
            success_rate = successful_pixels / len(processed_results) * 100

            print(f"结果保存至: {result_file}")
            print(f"成功率: {success_rate:.1f}% ({successful_pixels}/{len(processed_results)})")

            # AOD预测
            try:
                predict_AOD(result_file, base_dir)
                print("AOD预测完成")
            except Exception as e:
                print(f"AOD预测失败: {e}")

        else:
            print("没有获得有效结果")

    # 性能总结
    total_time = (time() - T_begin) / 60
    print(f"\n总处理时间: {total_time:.2f} 分钟")
    if 'total_coords' in locals():
        print(f"平均处理速度: {total_coords / total_time:.1f} 坐标/分钟")

    print("\n基于CPU绑定的大范围区域反演完成！")