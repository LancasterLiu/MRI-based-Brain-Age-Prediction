# data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(data_path, selected_regions=None, test_size=0.2, val_size=0.2, random_state=42,
              exclude_sex=False, target_col='age', feature_prefix='atlas_'):
    """
    加载并预处理脑龄预测数据。

    参数:
        data_path (str): CSV文件路径，应包含eid, age, sex以及以feature_prefix开头的体积列。
        selected_region_ids: list of int, 要保留的区域ID列表（例如 [1,2,3,...]）。
                             如果为None，则使用所有以 feature_prefix 开头的列。
        test_size (float): 测试集比例。
        val_size (float): 验证集比例（从训练集中划分）。
        random_state (int): 随机种子。
        exclude_sex (bool): 是否排除性别特征。
        target_col (str): 目标变量列名，通常是年龄。
        feature_prefix (str): 特征列的前缀，用于识别脑区体积列。

    返回:
        dict: 包含以下键值：
            'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test',
            'feature_names', 'scaler' (用于标准化的scaler对象)
    """
    # 读取数据
    df = pd.read_csv(data_path)
    print(f"原始数据形状: {df.shape}")

    # 检查必要列
    required_cols = ['eid', target_col, 'sex']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    # 提取特征列（所有以feature_prefix开头的列）
    if selected_regions is not None:
        # 根据选定的区域ID生成对应的列名，例如 'atlas_1', 'atlas_2', ...
        selected_cols = [f"{feature_prefix}{rid}" for rid in selected_regions if f"{feature_prefix}{rid}" in df.columns]
        feature_cols = selected_cols
    else:
        feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]

    if not feature_cols:
        raise ValueError(f"未找到以 '{feature_prefix}' 开头的特征列")
    print(f"找到 {len(feature_cols)} 个特征列")

    # 处理缺失值：删除目标变量缺失的行，特征缺失用均值填充
    df = df.dropna(subset=[target_col, 'sex']).copy()
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

    # 编码性别：假设sex为 'M'/'F' 或 0/1，统一转换为 0/1 (例如 Male=1, Female=0)
    if df['sex'].dtype == object:
        df['sex'] = df['sex'].map({'M': 1, 'F': 0, 'Male': 1, 'Female': 0})
    else:
        # 如果已经是数值，确保为0/1
        df['sex'] = df['sex'].astype(int)

    # 构建特征矩阵 X
    if not exclude_sex:
        X_cols = feature_cols + ['sex']
    else:
        X_cols = feature_cols
    X = df[X_cols].values
    y = df[target_col].values.ravel()

    # 划分数据集：先分出测试集，再从剩余中分出验证集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # 验证集大小相对于原始训练集的比例
    val_ratio_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, random_state=random_state
    )

    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

    # 特征标准化（基于训练集）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_val': X_val_scaled,
        'y_val': y_val,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'feature_names': X_cols,
        'scaler': scaler
    }

if __name__ == "__main__":
    # 简单测试
    data = load_data("./datasets/brain_features.csv")
    print("数据加载成功")