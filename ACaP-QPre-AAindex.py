import os
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
import argparse
import warnings
import time
import optuna
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
from logging.handlers import RotatingFileHandler

# 抑制特定环境警告以保持输出整洁
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 初始化输出目录与日志记录系统
OUTPUT_DIR = Path('/kaggle/working/MIC_prediction_outputs_aaindex_pca_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

fh = RotatingFileHandler(OUTPUT_DIR / 'training_pca_comparison.log', maxBytes=5 * 1024 * 1024, backupCount=5)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
logger.addHandler(ch)

# 定义数据与特征读取的全局路径

DATA_PATH = BASE_DIR / '/kaggle/input/datasets/leipengchong/wushap43/2246.csv'
AAINDEX_PATH = BASE_DIR / '/kaggle/input/wushap3/氨基酸特征'


# 从本地路径加载 566 维 AAindex 字典对象
def load_aaindex_features(path):
    aa_features_dict = {}
    path = Path(path)
    for file in path.glob('*.pkl'):
        try:
            features = joblib.load(file)
            if isinstance(features, dict):
                valid_features = {aa: value for aa, value in features.items() if
                                  aa in 'ACDEFGHIKLMNPQRSTVWY' and isinstance(value, (int, float))}
                if valid_features:
                    aa_features_dict[file.stem] = valid_features
        except Exception as e:
            logger.error(f"Error loading {file.name}: {e}", exc_info=True)
    return aa_features_dict


# 核心修改区：仅提取 566 维序列层级的 AAindex 特征
def compute_aaindex_only_features(seq, aa_features_dict):
    aaindex_features = []
    sorted_index_names = sorted(aa_features_dict.keys())

    for index_name in sorted_index_names:
        prop_dict = aa_features_dict[index_name]
        valid_values = [prop_dict[aa] for aa in seq if aa in prop_dict and prop_dict[aa] is not None]

        if valid_values:
            seq_prop_mean = np.mean(valid_values)
        else:
            seq_prop_mean = 0.0

        aaindex_features.append(seq_prop_mean)

    return aaindex_features


# 序列数据标准化清洗
def clean_sequence(seq, min_length=5, max_length=1000):
    if not isinstance(seq, str):
        return np.nan
    cleaned = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq.upper())
    if min_length <= len(cleaned) <= max_length:
        return cleaned
    return np.nan


# 定义模型性能评估体系 (包含所有6项指标)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)
    metrics = {
        'Test_MSE': mse,
        'Test_MAE': mae,
        'Test_R2': r2,
        'Test_Explained_Variance': evs,
        'Pearson_Correlation': pearson_corr,
        'Spearman_Correlation': spearman_corr
    }
    return metrics, y_pred


def plot_regression(y_true, y_pred, model_name, output_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, label='Predicted Values')
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_true))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel('True Values (Log MIC)')
    plt.ylabel('Predicted Values (Log MIC)')

    # 针对图片标题进行定制化映射
    if model_name == 'AAindex_NoPCA':
        display_title = 'AAindex: Predicted vs True Values'
    elif model_name == 'AAindex_WithPCA':
        display_title = 'AAindex(D): Predicted vs True Values'
    else:
        display_title = f'{model_name}: Predicted vs True Values'

    plt.title(display_title)
    plt.legend()
    plt.tight_layout()

    # 输出的文件命名保持不变，以确保文件系统兼容性
    plt.savefig(output_dir / f'{model_name}_pred_vs_true.png', dpi=300)
    plt.close()


# 构建通用实验框架
def run_experiment(experiment_name, use_pca, X_train, y_train, X_test, y_test, param_grid):
    logger.info(f"========== 开始执行实验: {experiment_name} ==========")
    print(f"\nRunning Experiment: {experiment_name}")

    steps = [('scaler', StandardScaler())]

    if use_pca:
        steps.append(('pca', PCA(n_components=0.95, random_state=42)))

    steps.append(('model', CatBoostRegressor(
        random_state=42,
        verbose=0,
        early_stopping_rounds=50,
        task_type='GPU',
        bootstrap_type='Poisson',
        devices='0'
    )))

    pipeline = Pipeline(steps)

    def optimize_catboost(trial, pipeline_opt, X, y):
        params = {}
        for key, values in param_grid.items():
            param_name = key.split('__')[1]
            if isinstance(values, list):
                params[key] = trial.suggest_categorical(param_name, values)
            elif isinstance(values, tuple) and len(values) == 2:
                params[key] = trial.suggest_float(param_name, values[0], values[1], log=True)
            else:
                params[key] = trial.suggest_int(param_name, values[0], values[1])

        pipeline_opt.set_params(**params)
        score = cross_val_score(pipeline_opt, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
        return -score.mean()

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: optimize_catboost(trial, pipeline, X_train, y_train), n_trials=50, timeout=3600)

    best_params = study.best_params
    pipeline.set_params(**{f'model__{k}': v for k, v in best_params.items()})
    pipeline.fit(X_train, y_train)

    if use_pca:
        n_components_retained = pipeline.named_steps['pca'].n_components_
        logger.info(f"PCA retained {n_components_retained} principal components to explain 95% variance.")
        print(f"PCA reduced dimensions from {X_train.shape[1]} to {n_components_retained}")

    # 独立测试集验证
    metrics, y_pred_test = evaluate_model(pipeline, X_test, y_test)

    # [新增] 全面输出所有的评估指标
    print(f"[{experiment_name}] Evaluation Metrics:")
    print(f"  - Test MSE:                 {metrics['Test_MSE']:.4f}")
    print(f"  - Test MAE:                 {metrics['Test_MAE']:.4f}")
    print(f"  - Test R^2:                 {metrics['Test_R2']:.4f}")
    print(f"  - Explained Variance:       {metrics['Test_Explained_Variance']:.4f}")
    print(f"  - Pearson Correlation:      {metrics['Pearson_Correlation']:.4f}")
    print(f"  - Spearman Correlation:     {metrics['Spearman_Correlation']:.4f}\n")

    logger.info(
        f"[{experiment_name}] Metrics -> MSE: {metrics['Test_MSE']:.4f}, MAE: {metrics['Test_MAE']:.4f}, R2: {metrics['Test_R2']:.4f}, ExpVar: {metrics['Test_Explained_Variance']:.4f}, Pearson: {metrics['Pearson_Correlation']:.4f}, Spearman: {metrics['Spearman_Correlation']:.4f}")

    joblib.dump(pipeline, OUTPUT_DIR / f'Model_{experiment_name}.pkl')
    plot_regression(y_test, y_pred_test, experiment_name, OUTPUT_DIR)

    return metrics


# 主执行模块
if __name__ == "__main__":
    start_total_time = time.time()

    logger.info("Reading and preprocessing data...")
    try:
        data = pd.read_csv(DATA_PATH, sep=',')
    except Exception as e:
        logger.error(f"Error reading data: {e}", exc_info=True)
        raise SystemExit(f"Failed to read data: {e}")

    q_low, q_high = data['MIC_Standardized'].quantile([0.01, 0.99])
    data['MIC_Standardized'] = data['MIC_Standardized'].clip(q_low, q_high)

    data['Sequence'] = data['Sequence'].apply(clean_sequence)
    data = data.dropna(subset=['Sequence'])
    data = data.reset_index(drop=True)

    logger.info(f"Number of valid sequences: {len(data['Sequence'])}")

    aa_features_dict = load_aaindex_features(AAINDEX_PATH)
    sequence_features = np.array([
        compute_aaindex_only_features(seq, aa_features_dict) for seq in
        tqdm(data['Sequence'], desc="Computing AAindex features")
    ], dtype=np.float32)

    X_feats = np.nan_to_num(sequence_features, nan=0.0, posinf=0.0, neginf=0.0)
    y = data['MIC_Standardized'].values

    indices = np.arange(len(data))
    X_train_main, X_test_main, y_train_main, y_test_main, _, _ = train_test_split(
        X_feats, y, indices, test_size=0.2, random_state=42
    )

    catboost_param_grid = {
        'model__iterations': [300, 500, 800],
        'model__depth': [4, 6, 8],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__l2_leaf_reg': [1, 3, 5],
        'model__bagging_temperature': [0.0, 0.5, 1.0],
        'model__grow_policy': ['SymmetricTree', 'Depthwise']
    }

    results = {}

    metrics_no_pca = run_experiment(
        experiment_name="AAindex_NoPCA",
        use_pca=False,
        X_train=X_train_main, y_train=y_train_main,
        X_test=X_test_main, y_test=y_test_main,
        param_grid=catboost_param_grid
    )
    results["No_PCA"] = metrics_no_pca

    metrics_pca = run_experiment(
        experiment_name="AAindex_WithPCA",
        use_pca=True,
        X_train=X_train_main, y_train=y_train_main,
        X_test=X_test_main, y_test=y_test_main,
        param_grid=catboost_param_grid
    )
    results["With_PCA"] = metrics_pca

    # [新增] 汇总阶段打印所有 6 种指标
    comparison_df = pd.DataFrame(results).T
    logger.info("========== 对比实验结果汇总 ==========")
    logger.info(f"\n{comparison_df.to_string()}")

    print("\n========== 对比实验结果汇总 (Comparative Results) ==========")
    # 强制重新排列列的顺序，使得输出更加符合学术审阅习惯
    ordered_columns = ['Test_MSE', 'Test_MAE', 'Test_R2', 'Test_Explained_Variance', 'Pearson_Correlation',
                       'Spearman_Correlation']
    print(comparison_df[ordered_columns].to_markdown())

    comparison_df.to_csv(OUTPUT_DIR / 'PCA_Ablation_Comparison.csv')

    total_training_time = time.time() - start_total_time
    logger.info(f"Total computing time: {total_training_time:.2f} seconds")