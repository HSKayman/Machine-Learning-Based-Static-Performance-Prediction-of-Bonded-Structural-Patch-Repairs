# Feature-importance / sensitivity analysis.
# Model-agnostic permutation importance is computed in the original input space
# (Material, thicknesses, temperature, patch type, crack length) rather than in the
# one-hot space, so the reported contributions correspond directly to the physical
# design variables. Importance is measured as the increase in MAPE when a single
# input column is randomly permuted.
#
# To stay consistent with the nested grouped protocol, importance is computed
# WITHIN the grouped outer folds: each model is trained on the augmented
# outer-train pool and the columns are permuted on the held-out outer-test group.
# The per-fold results are aggregated to a mean +/- std, so the reported values
# carry uncertainty and are not read off the training data.

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold

from models import make_model
from preprocess import DataPreprocessor, config_group_labels
from synthetic import augment_dataframe


def _feature_cols(df: pd.DataFrame):
    # Identify numeric, categorical, and target columns in a raw dataframe.
    pre = DataPreprocessor("tmp")
    float_cols, cat_cols, target_col, _ = pre._identify_columns(df)
    return float_cols, cat_cols, target_col


def permutation_importance(model, pre, eval_df, feature_cols, n_repeats=50, seed=42) -> Dict[str, float]:
    # Mean Δ MAPE when each original feature column is permuted on eval_df.
    X, y = pre.transform(eval_df)
    base = mean_absolute_percentage_error(y, model.predict(X)) * 100
    rng = np.random.RandomState(seed)
    importance = {}
    for col in feature_cols:
        deltas = []
        for _ in range(n_repeats):
            perturbed = eval_df.copy()
            perturbed[col] = rng.permutation(perturbed[col].values)
            Xp, _ = pre.transform(perturbed)
            deltas.append(mean_absolute_percentage_error(y, model.predict(Xp)) * 100 - base)
        importance[col] = float(np.mean(deltas))
    return importance


def run_dataset(dataset_type: str, config: Dict) -> None:
    # Grouped permutation importance (mean +/- std over outer folds) for the ANN
    # and the best overall model.
    output_dir = Path(config["data"]["output_dir"])
    baseline = pd.read_csv(output_dir / f"{dataset_type}_baseline.csv")
    groups = config_group_labels(baseline)
    with open(output_dir / f"{dataset_type}_best_configs.json") as f:
        best_configs = json.load(f)
    comp = pd.read_excel(output_dir / f"{dataset_type}_model_comparison.xlsx")

    float_cols, cat_cols, target_col = _feature_cols(baseline)
    feature_cols = float_cols + cat_cols

    syn_cfg = config.get("synthetic", {})
    target_samples = syn_cfg.get("target_samples", 300)
    factor = target_samples / max(1, len(baseline))
    ann_def = config["training"].get("ann", {})
    kan_def = config["training"].get("kan", {})
    rs = config.get("evaluation", {}).get("random_state", 42)
    n_repeats = config.get("evaluation", {}).get("importance_repeats", 50)
    k_outer = min(config.get("evaluation", {}).get("k_outer", 5), len(np.unique(groups)))

    best_overall = comp.sort_values("test_mape").iloc[0]["model"]
    models_to_explain = [n for n in dict.fromkeys(["ANN", best_overall]) if n in best_configs]

    print(f"\n{'=' * 60}\nFeature importance — {dataset_type}  "
          f"(grouped, {k_outer} folds, {n_repeats} permutations)\n{'=' * 60}")
    rows = []
    gkf = GroupKFold(n_splits=k_outer)
    for name in models_to_explain:
        per_fold = {c: [] for c in feature_cols}
        for tr_idx, te_idx in gkf.split(baseline, groups=groups):
            otr = baseline.iloc[tr_idx].reset_index(drop=True)
            ote = baseline.iloc[te_idx].reset_index(drop=True)
            aug, _ = augment_dataframe(otr, max(len(otr), int(round(len(otr) * factor))), syn_cfg)
            pre = DataPreprocessor(dataset_type).fit(aug)
            X_aug, y_aug = pre.transform(aug)
            model = make_model(name, best_configs[name], random_state=rs,
                               ann_defaults=ann_def, kan_defaults=kan_def)
            model.fit(X_aug, y_aug)
            imp = permutation_importance(model, pre, ote, feature_cols, n_repeats=n_repeats, seed=rs)
            for c, v in imp.items():
                per_fold[c].append(v)

        means = {c: float(np.mean(v)) for c, v in per_fold.items()}
        stds = {c: float(np.std(v)) for c, v in per_fold.items()}
        total = sum(abs(v) for v in means.values()) or 1.0
        print(f"\n  {name} (grouped permutation importance):")
        for col, v in sorted(means.items(), key=lambda kv: -kv[1]):
            print(f"    {col:28s} dMAPE={v:6.3f}% +/- {stds[col]:.3f}   ({100 * v / total:5.1f}%)")
            rows.append({"model": name, "feature": col, "delta_mape": v,
                         "delta_mape_std": stds[col], "relative_pct": 100 * v / total,
                         "n_permutations": n_repeats, "n_folds": k_outer})

    pd.DataFrame(rows).to_csv(output_dir / f"{dataset_type}_feature_importance.csv", index=False)
    print(f"\n  Saved {dataset_type}_feature_importance.csv")


def main():
    # CLI entry point for feature-importance analysis.
    parser = argparse.ArgumentParser(description="Feature importance / sensitivity analysis")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, choices=["patched", "unpatched", "both"], default="both")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    datasets = ["patched", "unpatched"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_dataset(ds, config)


if __name__ == "__main__":
    main()
