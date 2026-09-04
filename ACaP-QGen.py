# -*- coding: utf-8 -*-
"""
import pandas as pd
import numpy as np
import warnings
import os
import re
import shap
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools
from transformers import AutoTokenizer, AutoModel
from catboost import CatBoostRegressor
from tqdm import tqdm
from collections import Counter
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

# ==================== 1. 全局超参数与环境配置区域 ====================
INPUT_ROOT = Path('/kaggle/input')
OUTPUT_DIR = Path('/kaggle/working/AMP-PathA_RandomSelect')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ESM_DIR = INPUT_ROOT / 'zikan1'
MODEL_DIR = INPUT_ROOT / 'zikan2'
USER_DB_PATH = ESM_DIR / '2246.csv'
WEIGHTS_PATH = MODEL_DIR / 'train_best_model.pth'

TARGET_LEN = 12
ELITE_MIC_THRESHOLD = 10.0
CLS_FILTER_THRESHOLD = 0.99

# 热力学约束：Wimley-White 脂质-水分配标度字典
WW_SCALE = {
    'W': -1.85, 'F': -1.13, 'Y': -0.94, 'L': -0.56, 'I': -0.31,
    'C': -0.24, 'M': -0.23, 'G': 0.01, 'V': 0.07, 'S': 0.13,
    'T': 0.14, 'A': 0.17, 'H': 0.17, 'N': 0.42, 'P': 0.45,
    'Q': 0.58, 'R': 0.81, 'K': 0.99, 'E': 2.02, 'D': 2.07
}
# 多维理化参数边界条件
PHYS_LIMITS = {'charge': (3.0, 6.0), 'gravy': (-0.5, 0.8), 'ww': (-1.6, 2.0)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')


# ==================== 2. 深度学习模型架构定义 ====================
class ESModel(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.bert = AutoModel.from_pretrained(checkpoint_path, add_pooling_layer=False)
        self.projection = nn.Linear(640, 320, bias=True)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(320, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        input_ids = x['input_ids'].to(device)
        attention_mask = x['attention_mask'].to(device)
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            x = outputs.last_hidden_state[:, 0, :]
        x = self.projection(x)
        x = self.dropout(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.output_layer(x)
        return torch.softmax(x, dim=1)


class AFP_Predictor_Wrapper:
    def __init__(self, weights_path, checkpoint_dir):
        print(f"🚀 [初始化] 加载 ESM-2 高维特征提取器...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
        self.model = ESModel(checkpoint_path=str(checkpoint_dir))

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "").replace("bert.esm.", "bert.")
                if "bert.classifier.out_proj" in name:
                    name = name.replace("bert.classifier.out_proj", "projection")
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict, strict=False)
            print(f"✅ 模型参数映射与挂载完成。")
        self.model.to(device).eval()

    def predict_filter(self, sequences, threshold=0.99, batch_size=64):
        passed = []
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="ESM-2 特征甄别", leave=False):
                batch = sequences[i:i + batch_size]
                inputs = self.tokenizer(batch, max_length=30, padding='max_length',
                                        truncation=True, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                probs = self.model(inputs).cpu().numpy()[:, 1]
                for seq, p in zip(batch, probs):
                    if p >= threshold: passed.append(seq)
        return passed


# ==================== 3. 核心计算辅助函数 ====================
def compute_distance_matrix(sequences):
    n = len(sequences)
    dist_matrix = np.zeros((n, n))

    def levenshtein(s1, s2):
        if len(s1) < len(s2): return levenshtein(s2, s1)
        if len(s2) == 0: return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    for i in tqdm(range(n), desc="计算拓扑距离矩阵", leave=False):
        for j in range(i + 1, n):
            d = levenshtein(sequences[i], sequences[j])
            dist_matrix[i, j] = dist_matrix[j, i] = d
    return dist_matrix


def get_apaac_features(sequences, lambda_val=3, w=0.05):
    props = {'A': (0.62, -0.5), 'C': (0.29, -1.0), 'D': (-0.90, 3.0), 'E': (-0.74, 3.0),
             'F': (1.19, -2.5), 'G': (0.48, 0.0), 'H': (-0.40, -0.5), 'I': (1.38, -1.8),
             'K': (-1.50, 3.0), 'L': (1.06, -1.8), 'M': (0.64, -1.3), 'N': (-0.78, 0.2),
             'P': (0.12, 0.0), 'Q': (-0.85, 0.2), 'R': (-2.53, 3.0), 'S': (-0.18, 0.3),
             'T': (-0.05, -0.4), 'V': (1.08, -1.5), 'W': (0.81, -3.4), 'Y': (0.26, -2.3)}

    vals_h1 = np.array([v[0] for v in props.values()])
    vals_h2 = np.array([v[1] for v in props.values()])
    mean_h1, std_h1 = np.mean(vals_h1), np.std(vals_h1)
    mean_h2, std_h2 = np.mean(vals_h2), np.std(vals_h2)
    norm_props = {k: ((v[0] - mean_h1) / std_h1, (v[1] - mean_h2) / std_h2) for k, v in props.items()}
    aa_order = sorted(props.keys())

    features = []
    for seq in sequences:
        clean_seq = [aa for aa in seq if aa in norm_props]
        if len(clean_seq) < lambda_val + 1:
            features.append([0.0] * (20 + 2 * lambda_val))
            continue
        c = Counter(clean_seq)
        freqs = [c[aa] / len(clean_seq) for aa in aa_order]
        thetas = []
        for lag in range(1, lambda_val + 1):
            h1 = sum(
                norm_props[clean_seq[i]][0] * norm_props[clean_seq[i + lag]][0] for i in range(len(clean_seq) - lag))
            h2 = sum(
                norm_props[clean_seq[i]][1] * norm_props[clean_seq[i + lag]][1] for i in range(len(clean_seq) - lag))
            thetas.extend([h1 / (len(clean_seq) - lag), h2 / (len(clean_seq) - lag)])
        features.append(freqs + [w * t for t in thetas])
    return np.array(features)


# ==================== 4. 主流线 (Main Pipeline) ====================
def main_pipeline():
    # --- Step 1 & 2: 数据加载与 KFF (Key Feature Fragment) 提取 ---
    print("🔹 [阶段 1-2] 基线数据集加载与关键基序 (KFF) 提取...")
    if not USER_DB_PATH.exists(): return
    df = pd.read_csv(USER_DB_PATH)
    clean_df = df.dropna(subset=['Sequence', 'MIC']).reset_index(drop=True)
    clean_df['MIC'] = pd.to_numeric(clean_df['MIC'], errors='coerce')

    train_fragments = []
    for _, row in clean_df.iterrows():
        seq = re.sub(r'[^A-Z]', '', str(row['Sequence']).upper())
        if len(seq) < TARGET_LEN: continue
        for i in range(len(seq) - TARGET_LEN + 1):
            train_fragments.append({'Fragment': seq[i:i + TARGET_LEN], 'MIC': row['MIC']})
    frag_df = pd.DataFrame(train_fragments)

    reg_model = CatBoostRegressor(iterations=300, verbose=False, cat_features=list(range(TARGET_LEN))).fit(
        pd.DataFrame([list(s) for s in frag_df['Fragment']], columns=[f'P{i}' for i in range(TARGET_LEN)]),
        -np.log10(frag_df['MIC'].clip(0.01))
    )

    explainer = shap.TreeExplainer(reg_model)
    elite_kffs = []
    parents = clean_df[clean_df['MIC'] < ELITE_MIC_THRESHOLD]
    for _, row in tqdm(parents.iterrows(), total=len(parents), desc="解析KFF贡献度"):
        seq = re.sub(r'[^A-Z]', '', str(row['Sequence']).upper())
        if len(seq) < TARGET_LEN: continue
        sub_frags = [seq[i:i + TARGET_LEN] for i in range(len(seq) - TARGET_LEN + 1)]
        if not sub_frags: continue
        shap_sums = np.sum(explainer.shap_values(
            pd.DataFrame([list(s) for s in sub_frags], columns=[f'P{i}' for i in range(TARGET_LEN)])), axis=1)
        if shap_sums.max() > 0: elite_kffs.append(sub_frags[np.argmax(shap_sums)])
    elite_kffs = list(set(elite_kffs))
    print(f"    -> ✅ 获取高潜 KFF 数量: {len(elite_kffs)}")

    # --- Step 3: 强制二分系统发育分析与序列空间划分 (Forced Bisection) ---
    print(f"\n🔹 [阶段 3] 亚家族系统发育分析与强制双簇切分 (Subfamily A & B)...")

    # 1. 计算拓扑距离与构建系统发育树
    dist_matrix = compute_distance_matrix(elite_kffs)
    Z = linkage(squareform(dist_matrix), method='average')

    plt.figure(figsize=(10, 5))
    dendrogram(Z, no_labels=True)
    plt.title('Phylogenetic Tree of KFFs (Forced 2 Clusters)')
    plt.savefig(OUTPUT_DIR / 'phylogenetic_tree_evidence.png')
    plt.close()

    # 2. 强制层次聚类剪枝 (Forced Maxclust Pruning)
    cluster_labels = fcluster(Z, t=2, criterion='maxclust')

    # 临时收集簇数据
    subfamilies_raw = {i: [] for i in set(cluster_labels)}
    for i, label in enumerate(cluster_labels):
        subfamilies_raw[label].append(elite_kffs[i])

    # 3. 智能重命名：依据样本量将最大簇命名为 A，次之命名为 B
    sorted_subfamilies = sorted(subfamilies_raw.values(), key=len, reverse=True)
    subfamilies = {}
    if len(sorted_subfamilies) == 2:
        subfamilies['A'] = sorted_subfamilies[0]
        subfamilies['B'] = sorted_subfamilies[1]
    else:
        for idx, sub_list in enumerate(sorted_subfamilies):
            subfamilies[chr(65 + idx)] = sub_list

    print(f"✅ 根据系统发育树拓扑结构，强制划分为 {len(subfamilies)} 个亚家族。")

    all_gen_seqs_set = set()

    # 4. 解析各个亚家族并分别生成子空间
    for sub_id, kff_list in subfamilies.items():
        print(f"\n  ▶ [亚家族 {sub_id}]")
        print(f"    - 包含 KFF 数量: {len(kff_list)} 条 ({len(kff_list) / len(elite_kffs):.1%} of total)")

        pwm = []
        enriched_info = []
        for i in range(TARGET_LEN):
            counts = Counter([s[i] for s in kff_list])
            top3 = counts.most_common(3)
            pwm.append([k for k, v in top3])
            top3_str = ", ".join([f"{k} ({v / len(kff_list):.1%})" for k, v in top3])
            enriched_info.append(f"Pos{i + 1:02d}: {top3_str}")

        print(f"    - 亚家族特征 (各位点 Top3 共同富集氨基酸及占比):")
        for i in range(0, TARGET_LEN, 2):
            print(f"      {enriched_info[i]:<35} | {enriched_info[i + 1]}")

        sub_gen_seqs = [''.join(p) for p in itertools.product(*pwm)]
        print(f"    - 衍生的候选序列子空间规模: {len(sub_gen_seqs)}")
        all_gen_seqs_set.update(sub_gen_seqs)

    all_gen_seqs = list(all_gen_seqs_set)
    print(f"\n✅ 所有亚家族的序列子空间已合并去重，获得初始理论候选序列总数: {len(all_gen_seqs)}")

    # ==================== 数据持久化层 1：全量理论空间 ====================
    unfiltered_seqs_path = OUTPUT_DIR / 'Unfiltered_Sequences_from_Subfamilies.csv'
    pd.DataFrame({'Sequence': all_gen_seqs}).to_csv(unfiltered_seqs_path, index=False)
    print(f"📁 [数据归档] 成功导出未经洗脱的初始理论组合序列库。")
    print(f"   -> 已保存至: {unfiltered_seqs_path}")
    # ======================================================================

    # --- Step 4, 5: ESM 深度表征过滤与多维理化审计 ---
    print("\n🔹 [阶段 4-5] ESM表征甄别与理化边界约束 (包含 WW 疏水性标度)...")
    esm_filter = AFP_Predictor_Wrapper(WEIGHTS_PATH, ESM_DIR)
    passed = esm_filter.predict_filter(all_gen_seqs, threshold=CLS_FILTER_THRESHOLD)

    valid_seqs = []
    original_set = set(clean_df['Sequence'])
    for seq in passed:
        if seq in original_set: continue
        try:
            pa = ProteinAnalysis(seq)
            charge = pa.charge_at_pH(7.0)
            gravy = pa.gravy()
            ww_score = sum(WW_SCALE.get(aa, 0) for aa in seq)

            if (PHYS_LIMITS['charge'][0] <= charge <= PHYS_LIMITS['charge'][1] and
                    PHYS_LIMITS['gravy'][0] <= gravy <= PHYS_LIMITS['gravy'][1] and
                    PHYS_LIMITS['ww'][0] <= ww_score <= PHYS_LIMITS['ww'][1]):
                valid_seqs.append(seq)
        except:
            continue

    if len(valid_seqs) == 0:
        print("❌ 警告：无有效序列穿透多维过滤阈值。")
        return

    X_pred = pd.DataFrame([list(s) for s in valid_seqs], columns=[f'P{i}' for i in range(TARGET_LEN)])
    shap_vals = explainer.shap_values(X_pred)
    stable_seqs, stable_indices = [], []
    for i, seq in enumerate(valid_seqs):
        sv = shap_vals[i]
        bad_cnt, is_stable = 0, True
        for v in sv:
            if v < 0:
                bad_cnt += 1
            else:
                bad_cnt = 0
            if bad_cnt >= 3: is_stable = False; break
        if is_stable:
            stable_seqs.append(seq)
            stable_indices.append(i)

    # --- Step 6: 降维映射与无监督聚类 ---
    print("🔹 [阶段 6] 降维表征与 K-Means 聚类筛选...")
    if not stable_seqs: return

    X_stable_features = get_apaac_features(stable_seqs)
    X_scaled = StandardScaler().fit_transform(X_stable_features)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_

    X_stable_df = pd.DataFrame([list(s) for s in stable_seqs], columns=[f'P{i}' for i in range(TARGET_LEN)])
    pred_mics = 10 ** (-reg_model.predict(X_stable_df))
    df_final = pd.DataFrame({'Sequence': stable_seqs, 'Pred_MIC': pred_mics})

    df_final['PC1'] = coords[:, 0]
    df_final['PC2'] = coords[:, 1]

    n_final_k = 4 if len(df_final) >= 4 else 1
    kmeans = KMeans(n_clusters=n_final_k, random_state=None, n_init=10)
    df_final['Cluster'] = kmeans.fit_predict(coords)

    # ==================== 数据持久化层 2：有效新型序列全局输出 ====================
    all_seqs_path = OUTPUT_DIR / 'All_Newly_Designed_Sequences.csv'
    df_final.to_csv(all_seqs_path, index=False)
    print(f"\n📁 [数据归档] 成功提取所有穿透理化审计与表征预测的新型序列。")
    print(f"   -> 包含 {len(df_final)} 条序列，已保存至: {all_seqs_path}")
    # ==============================================================================

    final_cands_list = []
    df_final['Type'] = 'Other'
    TARGET_PER_CLUSTER = 20
    N_CENTER = 10
    N_EDGE = 10

    for c in range(n_final_k):
        cluster_indices = df_final[df_final['Cluster'] == c].index
        if len(cluster_indices) == 0: continue

        center = kmeans.cluster_centers_[c]
        features_2d = df_final.loc[cluster_indices, ['PC1', 'PC2']].values
        distances = np.linalg.norm(features_2d - center, axis=1)

        temp_df = df_final.loc[cluster_indices].copy()
        temp_df['Distance_to_Center'] = distances
        temp_df = temp_df.sort_values('Distance_to_Center')

        if len(temp_df) <= TARGET_PER_CLUSTER:
            df_final.loc[temp_df.index, 'Type'] = 'Center/Edge'
            final_cands_list.append(temp_df)
        else:
            centers = temp_df.head(N_CENTER).copy()
            df_final.loc[centers.index, 'Type'] = 'Center'
            edges = temp_df.tail(N_EDGE).copy()
            df_final.loc[edges.index, 'Type'] = 'Edge'

            final_cands_list.append(centers)
            final_cands_list.append(edges)

        print(f"     亚群 {c}: 遴选 {min(len(temp_df), TARGET_PER_CLUSTER)} 条代表性序列")

    final_cands = pd.concat(final_cands_list).drop_duplicates(subset=['Sequence'])

    plt.figure(figsize=(10, 8))
    plt.scatter(df_final['PC1'], df_final['PC2'], c=df_final['Cluster'], cmap='viridis', alpha=0.3, label='Candidates')

    markers = {'Center': ('red', '*', 150), 'Edge': ('blue', '^', 120), 'Center/Edge': ('purple', 'D', 120)}
    for t, (color, marker, size) in markers.items():
        subset = df_final[df_final['Sequence'].isin(final_cands['Sequence']) & (df_final['Type'] == t)]
        if not subset.empty:
            plt.scatter(subset['PC1'], subset['PC2'], c=color, s=size, marker=marker, edgecolors='k', label=t)

    plt.title(f'Geometry-Based Selection ({N_CENTER} Center + {N_EDGE} Edge)')
    plt.xlabel(f'PC1 ({var_ratio[0] * 100:.2f}% Explained)', fontsize=12)
    plt.ylabel(f'PC2 ({var_ratio[1] * 100:.2f}% Explained)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(OUTPUT_DIR / 'Step6_Expanded_Selection.png')

    out = final_cands[['Sequence', 'Pred_MIC', 'Cluster', 'Type', 'Distance_to_Center']]
    out.to_csv(OUTPUT_DIR / 'PathA_Final_Candidates_24.csv', index=False)
    print(f"\n🎉 最终代表性子集 ({len(out)} 条) 已存取:")
    print(out)


if __name__ == "__main__":
    main_pipeline()