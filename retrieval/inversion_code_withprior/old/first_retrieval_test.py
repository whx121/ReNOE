import numpy as np
import pandas as pd
import joblib



# ==============================================================================
#    在主程序启动时执行一次，而不是在函数内反复加载
# ==============================================================================
RF_MODEL_PATH = '/media/amers/SSD_part1/whx/ResNet_code/dynamic_prior/random_forest_aerosol_model.joblib'

VC_DEFAULT = 0.01  # 定义一个默认的无效值

try:
    rf_model = joblib.load(RF_MODEL_PATH)
except FileNotFoundError as e:
    print(f"错误：找不到模型文件！请确保模型文件存在。 {e}")
    rf_model = None

# ==============================================================================
# 2. 定义 first_retrieval 预测函数
# ==============================================================================

def first_retrieval(doy, lon, lat, elev, data_obs):
    """
    使用预训练的随机森林模型，从POLDER原始观测反演四种气溶胶组分的体积浓度。

    :param doy: Day of Year (int)
    :param lon: 经度 (float)
    :param lat: 纬度 (float)
    :param elev: 海拔 (float)    ！！！meter！！！
    :param data_obs: 一个二维 numpy 数组，形状为 (N_angles, 12)，
                     每一行代表一个角度的观测向量，顺序如下:
                     [sza, vza, phi, TOA443, TOA490, DolP490, TOA565,
                      TOA670, DolP670, TOA865, DolP865, TOA1020]
    :return: 一个包含四种气溶胶体积浓度的 numpy 数组，
             顺序为 [BC_vol, Urban_vol, SeaSalt_vol, Dust_vol]。
             如果失败，则返回充满无效值的数组。
    """
    if rf_model is None:
        print("错误：模型未加载，无法进行预测。")
        return np.full(4, VC_DEFAULT)

    # --- Step 1: 准备扁平化的特征向量 ---

    # 1.1 提取角度观测数据
    # 我们需要第2到第12个角度，对应 data_obs 的索引 2 到 12
    num_angles_to_extract = 11
    num_measurements_per_angle = 12

    angular_features = np.full((num_angles_to_extract, num_measurements_per_angle), np.nan)

    num_available_angles = data_obs.shape[0]

    # 安全地提取角度数据
    if num_available_angles > 2:
        # data_obs[1:] 是从第二个角度开始的所有数据
        # min(...) 确保我们不会超出实际拥有的角度数量
        num_to_copy = min(num_angles_to_extract, num_available_angles - 2)
        angular_features[:num_to_copy, :] = data_obs[2:2 + num_to_copy, :]

    # 1.2 将角度特征“压平”成一个一维向量
    flattened_angular_features = angular_features.flatten()

    # 1.3 组合基础特征和压平后的角度特征
    #    顺序必须与训练时完全一致！
    base_features = np.array([doy, lat, lon, elev])
    full_feature_vector = np.concatenate([base_features, flattened_angular_features])

    # 1.4 检查是否有NaN值。如果存在（比如有效角度<11个），模型可能无法处理
    if np.isnan(full_feature_vector).any():
        print("警告：输入特征向量中存在NaN值，可能由有效观测角度不足引起。")
        # 对于随机森林，某些实现可以处理NaN，但最好避免。
        # 这里我们选择返回无效值，也可以选择用0或平均值填充。
        return np.full(4, VC_DEFAULT)

    # 1.5 将一维向量转换为DataFrame，因为scikit-learn模型需要带特征名的输入
    #     首先，我们需要构建正确的特征名列表
    feature_columns = []
    feature_columns.extend(['DOY', 'lat', 'lon' , 'elev'])
    observation_names = [
        'sza', 'vza', 'phi', 'TOA443', 'TOA490', 'DolP490', 'TOA565',
        'TOA670', 'DolP670', 'TOA865', 'DolP865', 'TOA1020'
    ]
    for i in range(2, 13):
        for name in observation_names:
            feature_columns.append(f'{name}_ang{i}')

    # 创建只有一行的DataFrame
    X_input_df = pd.DataFrame([full_feature_vector], columns=feature_columns)


    try:

        component_volumes = rf_model.predict(X_input_df[feature_columns])[0]


        return component_volumes

    except Exception as e:
        print(f"RF预测过程中发生错误: {e}")
        return np.full(4, VC_DEFAULT)


import RFprior as rf
if __name__ == '__main__':
    file_path = '/media/amers/SSD_part1/whx/ResNet_code/retrieval/test_data/'
    file_name = 'POLDER3_L1B-BG1-030062M_2006-03-18T05-32-07_V1-01.h5'
    lin,col = 1013 ,4850
    data_obs, doy, cloud, elev, land_sea = rf.extract_polder_h5_(file_path, file_name, lin, col)
    print(data_obs)
    print(elev)


    mock_lon = 116.4
    mock_lat = 39.9


    print("\n--- 开始使用示例数据进行预测 ---")

    component_volumes = first_retrieval(
        doy=doy,
        lon=mock_lon,
        lat=mock_lat,
        elev=elev*1000,
        data_obs=data_obs
    )

    print("\n--- 预测结果 ---")
    if component_volumes[0] != VC_DEFAULT:
        result_labels = ['BC_vol', 'Urban_vol', 'SeaSalt_vol', 'Dust_vol']
        for label, value in zip(result_labels, component_volumes):
            print(f"  {label}: {value:.6f}")
    else:
        print("预测失败。")
