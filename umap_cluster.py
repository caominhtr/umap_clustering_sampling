import argparse
from collections import Counter
from pathlib import Path
from collections import defaultdict
import numpy as np
import optuna
import pandas as pd
import umap
from rdkit.Chem import AllChem, CanonSmiles, DataStructs, PandasTools
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# GENERATE FP

def generate_fpts(df, smiles_col="SMILES", rad=2, bits=1024):
    df = df.copy()
    df["Canon"] = df[smiles_col].apply(CanonSmiles)
    PandasTools.AddMoleculeColumnToFrame(df, "Canon", "Structure")
    df = df[~df['Structure'].isna()].reset_index(drop = True)
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, rad, bits) for mol in df["Structure"]]
    return pd.DataFrame(np.array(fps))


def calc_max_tanimoto(df1, df2):
    fps1 = [GetMorganFingerprintAsBitVect(m, 2, 2048) for m in df1["Structure"]]
    fps2 = [GetMorganFingerprintAsBitVect(m, 2, 2048) for m in df2["Structure"]]
    matrix = np.array([DataStructs.BulkTanimotoSimilarity(fp, fps2) for fp in fps1])
    return matrix.max()


# UMAP + CLUSTERING

def umap_optimize(df_morgan, n_trials=100, seed=42):
    n = df_morgan.shape[0] - 1
    max_value = max(2, round(n / 4))

    def objective(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 2, max_value)
        min_dist    = trial.suggest_float("min_dist", 0.0, 0.99)
        n_clusters  = trial.suggest_int("n_clusters", 2, max_value)

        embedding = umap.UMAP(
            n_components=2, metric="jaccard",
            n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed
        ).fit_transform(df_morgan)

        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embedding)
        return silhouette_score(embedding, labels)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=5)
    )
    study.optimize(objective, n_trials=n_trials)

    p = study.best_params

    embedding = umap.UMAP(
        n_components=2, metric="jaccard",
        n_neighbors=p["n_neighbors"], min_dist=p["min_dist"], random_state=seed
    ).fit_transform(df_morgan)

    labels = AgglomerativeClustering(n_clusters=p["n_clusters"]).fit_predict(embedding)
    return labels, embedding, p, study.best_value


# MERGE

def merge_similar_clusters(df, threshold=0.68):
    cluster_ids = df["Cluster"].unique()
    n = len(cluster_ids)

    sim_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            sim_mat[i, j] = sim_mat[j, i] = calc_max_tanimoto(
                df[df["Cluster"] == cluster_ids[i]].reset_index(drop=True),
                df[df["Cluster"] == cluster_ids[j]].reset_index(drop=True),
            )

    above = np.where(np.triu(sim_mat, k=1) >= threshold)
    B = pd.DataFrame({
        'A1': [int(cluster_ids[u]) for u in above[0]],
        'A2': [int(cluster_ids[v]) for v in above[1]]
    })

    final_list = []
    for i in B['A1'].unique():
        connections = B[B['A1'] == i]['A2'].tolist() + [i]
        final_list.append(set(connections))

    groups = []
    for lst in final_list:
        overlapping = [g for g in groups if not lst.isdisjoint(g)]
        groups = [g for g in groups if lst.isdisjoint(g)]
        for g in overlapping:
            lst.update(g)
        groups.append(lst)

    similar_ids = {c for g in groups for c in g}
    dissimilar_ids = [int(c) for c in cluster_ids if int(c) not in similar_ids]
    
    return groups, dissimilar_ids

# GREEDY ASSIGNMENT

def assign_folds(size_map: dict, n_folds: int) -> dict[int, list]:

    folds = {i: [] for i in range(n_folds)}
    sizes = {i: 0  for i in range(n_folds)}
    for label, count in sorted(size_map.items(), key=lambda x: x[1], reverse=True):
        target = min(sizes, key=sizes.get)
        folds[target].append(label)
        sizes[target] += count
    return folds


# PIPELINE

def run(input_csv, output_dir, n_folds=5, sim_threshold=0.68, n_trials=100, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    PandasTools.AddMoleculeColumnToFrame(df, "SMILES", "Structure")

    # Fingerprints + clustering
    df_morgan = generate_fpts(df)
    labels, embedding, best_params, best_score = umap_optimize(df_morgan, n_trials, seed)
    df["Cluster"] = labels

    # Merge similar clusters
    similar_groups, dissimilar_ids = merge_similar_clusters(df, sim_threshold)
    counts = Counter(df["Cluster"])

    # Fold assignment for similar and dissimilar clusters separately
    sim_sizes  = {f"Sim_{i}": sum(counts[c] for c in g) for i, g in enumerate(similar_groups)}
    diff_sizes = {c: counts[c] for c in dissimilar_ids}

    sim_folds  = assign_folds(sim_sizes,  n_folds)
    diff_folds = assign_folds(diff_sizes, n_folds)

    # Map cluster to fold
    cluster_to_fold = {}
    for fold_idx, labels_ in sim_folds.items():
        for lbl in labels_:
            idx = int(lbl.split("_")[1])
            for cid in similar_groups[idx]:
                cluster_to_fold[cid] = fold_idx
    for fold_idx, cluster_ids in diff_folds.items():
        for cid in cluster_ids:
            cluster_to_fold[cid] = fold_idx

    df["Fold"] = df["Cluster"].map(cluster_to_fold)

    # Save outputs
    safe_cols = [c for c in df.columns if c not in ("Structure", "Canon")]
    df[safe_cols].to_csv(output_dir / "fold_detail_clustered.csv", index=False)

    for fold_idx in range(n_folds):
        fold_df = df[df["Fold"] == fold_idx][["ID", "SMILES"]]
        fold_df.to_csv(f'{output_dir}/fold_{fold_idx+1}.txt', index = False)

    fold_mat = np.zeros((n_folds, n_folds))
    for i in range(n_folds):
        for j in range(i):
            fold_mat[i, j] = fold_mat[j, i] = calc_max_tanimoto(
                df[df["Fold"] == i].reset_index(drop=True),
                df[df["Fold"] == j].reset_index(drop=True),
            )
 
    np.savetxt(output_dir/"maximum_tanimoto.txt", fold_mat, fmt="%.4f")

    with open(output_dir/"best_params.txt", "w") as f:
        f.write(f"best_score (silhouette): {best_score:.4f}\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")

        for i in range(n_folds):
            f.write(f"Fold {i+1}: {df[df["Fold"] == i].shape[0]}\n")



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV with ID and SMILES columns")
    p.add_argument("--output", default="results/")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.68)
    p.add_argument("--seed",type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_csv=args.input,
        output_dir=args.output,
        n_folds=args.folds,
        sim_threshold=args.threshold,
        n_trials=args.trials,
        seed=args.seed,
    )