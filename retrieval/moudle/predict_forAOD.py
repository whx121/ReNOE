'''
    读取最优化结果，进行AOD AAOD SSA 预测
'''
import numpy as np
import pandas as pd
import os
import joblib
from retrieval.moudle.DNNModel import DNNModel
import torch
import time


def predict_AOD_AAOD_SSA(OE_result_file, model_para_dir):
    """
    读取反演文件中的气溶胶体积浓度和ALH，使用已训练的PyTorch神经网络预测AOD、AAOD和SSA，
    并将预测结果追加写入原CSV文件中。
    """
    # ----------------------
    # 1. 加载模型与缩放器
    # ----------------------
    state_vector_cols = ["vc_BB", "vc_Urban", "vc_Ocean", "vc_Dust"]

    model_aerosol = DNNModel.load_model(os.path.join(model_para_dir, 'dnn_model_aerosol.pth'))
    scaler_x_aerosol = joblib.load(os.path.join(model_para_dir, 'scaler_features_aerosol.pkl'))
    scaler_y_aerosol = joblib.load(os.path.join(model_para_dir, 'scaler_y_aerosol.pkl'))

    # ----------------------
    # 2. 读取文件与数据清洗
    # ----------------------
    OE_result = pd.read_csv(OE_result_file)

    # 保留输入特征中的正常数值行
    valid_idx = np.isfinite(OE_result[state_vector_cols].values).all(axis=1)
    OE_result = OE_result[valid_idx].copy()

    if OE_result.shape[0] == 0:
        raise ValueError("输入数据经过清洗后为空（全为NaN/inf）。请检查OE_result_file。")

    # ----------------------
    # 3. 多输出预测
    # ----------------------
    x_scaled_aerosol = scaler_x_aerosol.transform(OE_result[state_vector_cols])
    y_pred_aerosol_scaled = model_aerosol.predict(x_scaled_aerosol)
    y_pred_aerosol = scaler_y_aerosol.inverse_transform(y_pred_aerosol_scaled)

    # 按照模型输出顺序分配结果
    # ["AOD_560nm", "SSA_560nm", "AAOD_560nm", "mr", "mi", "aod_fine", "aod_coarse"]
    OE_result['AOD'] = y_pred_aerosol[:, 0]
    OE_result['SSA'] = y_pred_aerosol[:, 1]
    OE_result['AAOD'] = y_pred_aerosol[:, 2]
    OE_result['MR'] = y_pred_aerosol[:, 3]
    OE_result['MI'] = y_pred_aerosol[:, 4]
    OE_result['fineAOD'] = y_pred_aerosol[:, 5]
    OE_result['coarseAOD'] = y_pred_aerosol[:, 6]

    # ----------------------
    # 4. 保存
    # ----------------------
    OE_result.to_csv(OE_result_file, index=False)
    print(f"预测结果已追加写入：{OE_result_file}")

