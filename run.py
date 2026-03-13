# run.py for train_test
import argparse
import os
import time
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import torch
from tqdm import tqdm
from data_loader import load_data
from model import BrainAgeModel
from sklearn.model_selection import GridSearchCV
import joblib
import json

def get_args():
    parser = argparse.ArgumentParser(description="脑龄预测模型训练与测试")
    parser.add_argument('--data_path', type=str, default='./datasets/brain_features.csv', help='特征CSV文件路径')
    parser.add_argument('--train', action='store_true', default=True, help='是否训练模型')
    parser.add_argument('--test', action='store_true', default=True, help='是否测试模型')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--model_type', type=str, default='lasso', choices=['lasso', 'svr', 'elasticnet', 'cnn'],
                        help='模型类型')
    # parser.add_argument('--output_model', type=str, default='./models/brain_age_model.pkl', help='输出模型文件路径')
    # parser.add_argument('--output_metrics', type=str, default='./results/metrics.json', help='输出评估指标JSON文件')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.2, help='验证集比例（从训练集中划分）')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    parser.add_argument('--tune', action='store_true', help='是否在验证集上调参（使用网格搜索）')
    parser.add_argument('--exclude_sex', action='store_true', default=True, help='是否排除性别特征')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    # save paths
    runtime=time.strftime("%Y-%m-%d_%H-%M")
    output_model = os.path.join('./models', args.model_type, runtime, 'brain_age_model.pkl')
    os.makedirs(os.path.dirname(output_model), exist_ok=True)

    output_metrics = os.path.join('./results', args.model_type, runtime, 'metrics.json')
    os.makedirs(os.path.dirname(output_metrics), exist_ok=True)

    # 1. 加载数据
    print("加载数据...")
    my_119_regions = None #[1,2,3,...] 指定要使用的区域ID列表，或设置为None使用所有特征
    data = load_data(
        data_path=args.data_path,
        selected_regions=my_119_regions,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        exclude_sex=args.exclude_sex
    )
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    scaler = data['scaler']

    # 2. 模型定义与超参数设置
    input_dim = data['X_train'].shape[1]
    if args.tune:
        print("执行超参数网格搜索...")
        if args.model_type == 'lasso':
            param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0]}
            base_model = BrainAgeModel(model_type='lasso')
        elif args.model_type == 'svr':
            param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
            base_model = BrainAgeModel(model_type='svr')
        elif args.model_type == 'elasticnet':
            param_grid = {'alpha': [0.001, 0.01, 0.1], 'l1_ratio': [0.2, 0.5, 0.8]}
            base_model = BrainAgeModel(model_type='elasticnet')
        else:
            raise ValueError("不支持的模型类型")

        # 使用GridSearchCV在训练集上交叉验证（这里为了简单，我们直接在训练集上5折）
        grid_search = GridSearchCV(base_model.model, param_grid, cv=5, scoring='neg_mean_absolute_error')
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print("最佳参数:", best_params)
        # 用最佳参数创建最终模型
        model = BrainAgeModel(model_type=args.model_type, **best_params)
    else:
        # 使用默认参数
        model = BrainAgeModel(model_type=args.model_type, input_dim=input_dim, epochs=args.epochs)

    # 3. 训练模型
    if args.train:
        print("训练模型...")
        model.fit(X_train, y_train)
        train_metrics = model.evaluate(X_train, y_train)

        # 4. 验证集评估（可选调参后）
        val_metrics = model.evaluate(X_val, y_val)
        print(f"\n验证集 MAE: {val_metrics['MAE']:.4f}, R²: {val_metrics['R2']:.4f}")

    # 5. 测试集评估（未校正偏差）
    if args.test:
        test_metrics_raw = model.evaluate(X_test, y_test)
        print(f"\n测试集 (原始) MAE: {test_metrics_raw['MAE']:.4f}, R²: {test_metrics_raw['R2']:.4f}")

        # 6. 年龄偏差校正
        # 在训练集上拟合校正参数
        y_train_pred = model.predict(X_train)
        y_train_corrected = model.age_bias_correction(y_train, y_train_pred)  # 实际返回校正后的值，同时内部拟合了线性模型
        # 注意：age_bias_correction 方法返回校正后的预测值，但我们需要保存校正器以便在测试集上使用
        # 为了简单，我们单独实现校正逻辑
        from sklearn.linear_model import LinearRegression
        bias_corrector = LinearRegression()
        bias_corrector.fit(y_train.reshape(-1, 1), y_train_pred)
        # 对测试集进行校正
        y_test_pred = model.predict(X_test)
        bias_test = bias_corrector.predict(y_test.reshape(-1, 1)) - y_test_pred
        y_test_corrected = y_test_pred + bias_test
        # 计算校正后的MAE
        from sklearn.metrics import mean_absolute_error
        mae_corrected = mean_absolute_error(y_test, y_test_corrected)
        r2_corrected = r2_score(y_test, y_test_corrected)
        print(f"测试集 (校正后) MAE: {mae_corrected:.4f}, R²: {r2_corrected:.4f}")

    # 7. 保存模型和scaler
    model.save(output_model)
    # 同时保存scaler和bias_corrector，以便后续对新数据预测
    joblib.dump({
        'scaler': scaler,
        'bias_corrector': bias_corrector
    }, output_model.replace('.pkl', '_preprocess.pkl'))

    # 8. 保存指标
    metrics = {
        'val': val_metrics,
        'test_raw': test_metrics_raw,
        'test_corrected': {'MAE': mae_corrected, 'R2': r2_corrected},
        'best_params': best_params if args.tune else None
    }
    with open(output_metrics, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"指标已保存至 {output_metrics}")

if __name__ == "__main__":
    main()