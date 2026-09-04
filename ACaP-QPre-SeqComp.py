# 2246-F为降维
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
import warnings
import time
import optuna
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
from logging.handlers import RotatingFileHandler

# 抑制特定环境警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 初始化输出目录与日志系统
OUTPUT_DIR = Path('/kaggle/working/MIC_prediction_outputs_2246F_ablation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

fh = RotatingFileHandler(OUTPUT_DIR / 'training_2246F_ablation.log', maxBytes=5 * 1024 * 1024, backupCount=5)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
logger.addHandler(ch)

# 全局路径：本次实验完全只需这一个文件
FEATURE_CSV_PATH = Path('/kaggle/input/datasets/leipengchong/wushap43/2246-F.CSV')


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)
    return {
        'Test_MSE': mse,
        'Test_MAE': mae,
        'Test_R2': r2,
        'Test_Explained_Variance': evs,
        'Pearson_Correlation': pearson_corr,
        'Spearman_Correlation': spearman_corr
    }, y_pred


def plot_regression(y_true, y_pred, model_name, output_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, label='Predicted Values')
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_true))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel('True Values (Log MIC)')
    plt.ylabel('Predicted Values (Log MIC)')
    plt.title(f'{model_name}: Predicted vs True (2246-F Only)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_pred_vs_true.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    start_total_time = time.time()

    # 阶段 1：直接加载 2246-F 宽表
    logger.info("Loading 2246-F feature matrix...")
    print("Loading 2246-F data...")
    try:
        data = pd.read_csv(FEATURE_CSV_PATH, sep=',')
    except Exception as e:
        logger.error(f"Error reading feature database: {e}", exc_info=True)
        raise SystemExit(f"Failed to read file: {e}")

    logger.info(f"Loaded dataset shape: {data.shape}")
    print(f"Loaded dataset shape: {data.shape}")

    # 阶段 2：提取 Target 并清洗异常值
    q_low, q_high = data['MIC_Standardized'].quantile([0.01, 0.99])
    data['MIC_Standardized'] = data['MIC_Standardized'].clip(q_low, q_high)
    y = data['MIC_Standardized'].values

    # 阶段 3：严格拦截数据泄露 (Data Leakage Block)
    logger.info("Parsing features and blocking target leakage...")

    feature_names = []
    for col in data.columns:
        col_lower = str(col).lower()
        # 剔除元数据
        if col_lower in ['sequence', 'id', 'unnamed: 0']:
            continue
        # 核心防线：凡是名字里带 'mic' 的列全部丢弃，绝不喂给特征矩阵
        if 'mic' in col_lower:
            continue
        feature_names.append(col)

    X_feats = data[feature_names].values
    logger.info(f"Safely extracted feature space shape: {X_feats.shape} ({len(feature_names)} features)")
    print(f"Safely extracted feature space shape: {X_feats.shape} (Leakage Strictly Blocked)")

    # 清洗 NaN
    X_feats = np.nan_to_num(X_feats, nan=0.0, posinf=0.0, neginf=0.0)

    # 阶段 4：数据集划分
    indices = np.arange(len(data))
    X_train_main, X_test_main, y_train_main, y_test_main, train_indices, test_indices = train_test_split(
        X_feats, y, indices, test_size=0.2, random_state=42
    )

    # 阶段 5：超参数网格与管道配置
    catboost_param_grid = {
        'model__iterations': [200, 300, 400, 500, 600, 700, 800],
        'model__depth': [4, 6, 8, 10],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'model__l2_leaf_reg': [1, 3, 5, 7, 9],
        'model__bagging_temperature': [0.0, 0.2, 0.5, 0.8],
        'model__border_count': [64, 128, 256],
        'model__grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
        'model__bootstrap_type': ['Poisson'],
        'model__subsample': [0.6, 0.8, 0.9, 1.0]
    }

    pipeline = Pipeline([
        ('feature_selection', SelectKBest(mutual_info_regression, k=800)),
        ('model', CatBoostRegressor(
            random_state=42,
            verbose=0,
            early_stopping_rounds=50,
            task_type='GPU',
            bootstrap_type='Poisson',
            devices='0'
        ))
    ])


    def optimize_catboost(trial, pipeline, param_grid, X, y):
        params = {}
        for key, values in param_grid.items():
            param_name = key.split('__')[1]
            if isinstance(values, list):
                params[key] = trial.suggest_categorical(param_name, values)
            elif isinstance(values, tuple) and len(values) == 2:
                params[key] = trial.suggest_float(param_name, values[0], values[1], log=True)
            else:
                params[key] = trial.suggest_int(param_name, values[0], values[1])
        pipeline.set_params(**params)
        score = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
        return -score.mean()


    # 阶段 6：模型优化与拟合
    logger.info("Starting Optuna hyperparameter optimization...")
    print("Starting Optuna hyperparameter optimization...")
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner())
    # 缩减 trial 数量以加快验证速度，如果需要可修改为 150
    study.optimize(lambda trial: optimize_catboost(trial, pipeline, catboost_param_grid, X_train_main, y_train_main),
                   n_trials=100, timeout=7200)

    best_params = study.best_params
    pipeline.set_params(**{f'model__{k}': v for k, v in best_params.items()})

    logger.info("Fitting the final ablation model...")
    pipeline.fit(X_train_main, y_train_main)

    selector = pipeline.named_steps['feature_selection']
    scores = selector.scores_

    feature_scores_df = pd.DataFrame({
        'Feature_Name': feature_names,
        'Score': scores
    })
    feature_scores_df = feature_scores_df.sort_values(by='Score', ascending=False)
    feature_scores_df.to_csv(OUTPUT_DIR / '2246F_feature_scores.csv', index=False)

    # 阶段 7：独立验证
    metrics, y_pred_test = evaluate_model(pipeline, X_test_main, y_test_main)

    print("\n================ FINAL RESULTS ================")
    print(f"  2246-F Ablation Test MSE: {metrics['Test_MSE']:.3f}")
    print(f"  2246-F Ablation Test MAE: {metrics['Test_MAE']:.3f}")
    print(f"  2246-F Ablation Test R^2: {metrics['Test_R2']:.3f}")
    print("===============================================\n")

    model_path = OUTPUT_DIR / 'CatBoost_2246F_Ablation.pkl'
    joblib.dump(pipeline, model_path)

    plot_regression(y_test_main, y_pred_test, 'CatBoost_2246F_Ablation', OUTPUT_DIR)
    plot_residuals(y_test_main, y_pred_test, 'CatBoost_2246F_Ablation', OUTPUT_DIR)

    end_time_total = time.time()
    total_training_time = end_time_total - start_total_time
    print(f"Ablation Study completed safely. Total computing time: {total_training_time:.2f} seconds")