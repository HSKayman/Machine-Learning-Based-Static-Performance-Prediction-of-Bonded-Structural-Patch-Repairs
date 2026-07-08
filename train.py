# PyTorch training module with hyperparameter grid search.

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

from models import DEVICE, get_param_grids, make_model
from preprocess import DataPreprocessor
from synthetic import augment_dataframe

warnings.filterwarnings("ignore")


def _material_labels(df: pd.DataFrame) -> np.ndarray:
    mat = next((c for c in df.columns if c.strip().lower() == "material"), None)
    return df[mat].astype(str).values if mat else np.zeros(len(df))


def _metrics(y_true, y_pred) -> Tuple[float, float, float]:
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mape, mse, r2


def build_fold_data(train_df: pd.DataFrame, dataset_type: str, config: Dict[str, Any], factor: float):
    # Precompute, ONCE, the augmented and baseline-only arrays for every CV fold.
    #
    # This keeps the protocol leakage-free (the GMM and preprocessing are fit on the
    # training partition of each fold only) while avoiding recomputing augmentation
    # for every model/config, which is the expensive part.
    # Returns (aug_folds, base_folds), each a list of (X_tr, y_tr, X_va, y_va).
    #
    eval_cfg = config.get("evaluation", {})
    k = eval_cfg.get("k_folds", 5)
    rs = eval_cfg.get("random_state", 42)
    syn_cfg = config.get("synthetic", {})

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rs)
    labels = _material_labels(train_df)

    aug_folds, base_folds = [], []
    for tr_idx, va_idx in skf.split(train_df, labels):
        tr = train_df.iloc[tr_idx].reset_index(drop=True)
        va = train_df.iloc[va_idx].reset_index(drop=True)

        target = max(len(tr), int(round(len(tr) * factor)))
        tr_aug, _ = augment_dataframe(tr, target, syn_cfg)
        pre_a = DataPreprocessor(dataset_type).fit(tr_aug)
        aug_folds.append((*pre_a.transform(tr_aug), *pre_a.transform(va)))

        pre_b = DataPreprocessor(dataset_type).fit(tr)
        base_folds.append((*pre_b.transform(tr), *pre_b.transform(va)))
    return aug_folds, base_folds


def build_external_data(train_df, test_df, dataset_type, config, factor):
    # Augmented full training pool + transformed real hold-out test set (fit once).
    syn_cfg = config.get("synthetic", {})
    target = max(len(train_df), int(round(len(train_df) * factor)))
    tr_aug, _ = augment_dataframe(train_df, target, syn_cfg)
    pre = DataPreprocessor(dataset_type).fit(tr_aug)
    X_tr, y_tr = pre.transform(tr_aug)
    X_te, y_te = pre.transform(test_df)
    return X_tr, y_tr, X_te, y_te, pre


def cv_score(folds, model_name, params, config) -> Dict[str, float]:
    # Train/evaluate a model over precomputed folds. Validation is real data only.
    rs = config.get("evaluation", {}).get("random_state", 42)
    ann_def = config["training"].get("ann", {})
    kan_def = config["training"].get("kan", {})
    mapes, r2s = [], []
    for X_tr, y_tr, X_va, y_va in folds:
        model = make_model(model_name, params, random_state=rs, ann_defaults=ann_def, kan_defaults=kan_def)
        model.fit(X_tr, y_tr)
        mape, _, r2 = _metrics(y_va, model.predict(X_va))
        mapes.append(mape)
        r2s.append(r2)
    return {
        "mean_mape": float(np.mean(mapes)), "std_mape": float(np.std(mapes)),
        "mean_r2": float(np.mean(r2s)), "std_r2": float(np.std(r2s)),
    }


def external_score(ext, model_name, params, config):
    # Train on the augmented training pool, evaluate on the real hold-out test set.
    rs = config.get("evaluation", {}).get("random_state", 42)
    ann_def = config["training"].get("ann", {})
    kan_def = config["training"].get("kan", {})
    X_tr, y_tr, X_te, y_te, pre = ext
    model = make_model(model_name, params, random_state=rs, ann_defaults=ann_def, kan_defaults=kan_def)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mape, _, r2 = _metrics(y_te, pred)
    return mape, r2, model, pre, pred


def run_dataset(dataset_type: str, config: Dict[str, Any]) -> None:
    output_dir = Path(config["data"]["output_dir"])
    baseline = pd.read_csv(output_dir / f"{dataset_type}_baseline.csv")
    train_df = pd.read_csv(output_dir / f"{dataset_type}_baseline_train.csv")
    test_df = pd.read_csv(output_dir / f"{dataset_type}_baseline_test.csv")

    target_samples = config.get("synthetic", {}).get("target_samples", 300)
    factor = target_samples / max(1, len(baseline))  # e.g. 300/75 = 4x

    print(f"\n{'=' * 62}\nDataset: {dataset_type}  |  baseline={len(baseline)} "
          f"train={len(train_df)} test={len(test_df)}  aug factor={factor:.2f}x\n{'=' * 62}")

    print("  Precomputing fold data (leakage-free augmentation, once per fold)...")
    aug_folds, base_folds = build_fold_data(train_df, dataset_type, config, factor)
    ext = build_external_data(train_df, test_df, dataset_type, config, factor)

    grids = get_param_grids(config["training"])
    all_rows: List[Dict[str, Any]] = []

    # ---- Grid search (selection CV, augmented, leakage-free) ----
    for model_name, param_list in grids.items():
        print(f"\n  [{model_name}] {len(param_list)} configs")
        t0 = time.time()
        for i, params in enumerate(param_list):
            try:
                res = cv_score(aug_folds, model_name, params, config)
                all_rows.append({"model": model_name, "params": json.dumps(params), **res})
                if (i + 1) % max(1, len(param_list) // 4) == 0 or len(param_list) <= 6:
                    print(f"    [{i+1}/{len(param_list)}] MAPE={res['mean_mape']:.2f}% "
                          f"R2={res['mean_r2']:.3f}")
            except Exception as e:
                print(f"    [{i+1}/{len(param_list)}] ERROR: {e}")
        print(f"    ({time.time() - t0:.1f}s)")

    all_df = pd.DataFrame(all_rows).sort_values("mean_mape").reset_index(drop=True)
    all_df.to_excel(output_dir / f"{dataset_type}_all_results.xlsx", index=False)

    # Full ANN grid for the appendix.
    ann_df = all_df[all_df["model"] == "ANN"].copy().reset_index(drop=True)
    if not ann_df.empty:
        ann_df.insert(0, "rank", np.arange(1, len(ann_df) + 1))
        ann_df.to_excel(output_dir / f"{dataset_type}_ann_grid.xlsx", index=False)

    # ---- Best config per model: baseline-only CV + external test ----
    comparison, best_configs = [], {}
    test_predictions = {"y_true": test_df[_target_col(test_df)].values,
                        "source": _source_vals(test_df)}
    for model_name in grids:
        sub = all_df[all_df["model"] == model_name]
        if sub.empty:
            continue
        best = sub.iloc[0]
        params = json.loads(best["params"])
        best_configs[model_name] = params

        base_cv = cv_score(base_folds, model_name, params, config)
        test_mape, test_r2, model, pre, pred = external_score(ext, model_name, params, config)

        comparison.append({
            "model": model_name,
            "params": json.dumps(params),
            "cv_aug_mape": best["mean_mape"], "cv_aug_mape_std": best["std_mape"],
            "cv_aug_r2": best["mean_r2"], "cv_aug_r2_std": best["std_r2"],
            "cv_baseline_mape": base_cv["mean_mape"], "cv_baseline_r2": base_cv["mean_r2"],
            "test_mape": test_mape, "test_r2": test_r2,
        })
        test_predictions[f"pred_{model_name}"] = pred
        print(f"  -> {model_name:16s} cvAug={best['mean_mape']:.2f}%  "
              f"cvBase={base_cv['mean_mape']:.2f}%  test={test_mape:.2f}% (R2={test_r2:.3f})")

        # Save the ANN (primary model) retrained artifact + preprocessor.
        if model_name == "ANN":
            import torch
            torch.save({"model_state_dict": model.model.state_dict(),
                        "config": model.get_config(), "input_dim": model.input_dim},
                       output_dir / f"{dataset_type}_final_ann.pt")
            with open(output_dir / f"{dataset_type}_final_preprocessor.pkl", "wb") as f:
                pickle.dump(pre.__dict__, f)

    comp_df = pd.DataFrame(comparison).sort_values("test_mape").reset_index(drop=True)
    comp_df.to_excel(output_dir / f"{dataset_type}_model_comparison.xlsx", index=False)
    pd.DataFrame(test_predictions).to_csv(output_dir / f"{dataset_type}_test_predictions.csv", index=False)
    with open(output_dir / f"{dataset_type}_best_configs.json", "w") as f:
        json.dump(best_configs, f, indent=2)

    print(f"\n  === Model comparison ({dataset_type}) — sorted by hold-out test MAPE ===")
    for _, r in comp_df.iterrows():
        print(f"    {r['model']:16s}  test MAPE={r['test_mape']:5.2f}%  R2={r['test_r2']:.3f}  "
              f"| cv(aug)={r['cv_aug_mape']:.2f}%  cv(base)={r['cv_baseline_mape']:.2f}%")


def _target_col(df):
    return next((c for c in df.columns if "failure" in c.lower() or "load" in c.lower()), df.columns[-1])


def _source_vals(df):
    c = next((c for c in df.columns if c.lower() == "source"), None)
    return df[c].values if c else np.array(["baseline"] * len(df))


def main():
    parser = argparse.ArgumentParser(description="Leakage-free training and model comparison")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, choices=["patched", "unpatched", "both"], default="both")
    parser.add_argument("--use-augmented", action="store_true", default=True)  # kept for orchestrator compat
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Using device: {DEVICE}")
    datasets = ["patched", "unpatched"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_dataset(ds, config)


if __name__ == "__main__":
    main()
