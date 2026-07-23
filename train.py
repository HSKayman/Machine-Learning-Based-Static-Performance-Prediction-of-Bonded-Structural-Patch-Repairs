# Training module with nested grouped cross-validation.
#
# Protocol (leakage-free and grouped by physical configuration):
#   * groups = unique physical configuration (see preprocess.config_group_labels),
#     so replicates, FE, and theoretical rows of the same design point never split
#     across folds. This measures generalization to unseen configurations, not
#     specimen-to-specimen scatter.
#   * Model family + hyperparameters are chosen inside an INNER GroupKFold loop;
#     the OUTER GroupKFold loop is used only for evaluation. The reported test
#     metric is therefore an unbiased nested-CV estimate, not a hold-out that was
#     also used for model selection.
#   * The same outer folds are scored with GMM augmentation ON and OFF (paired),
#     so the benefit (or not) of the synthetic data can be reported honestly.

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
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from models import DEVICE, get_param_grids, make_model
from preprocess import DataPreprocessor, config_group_labels
from synthetic import augment_dataframe

warnings.filterwarnings("ignore")


def _metrics(y_true, y_pred) -> Tuple[float, float, float]:
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mape, mse, r2


def _target_col(df):
    return next((c for c in df.columns if "failure" in c.lower() or "load" in c.lower()), df.columns[-1])


def _source_vals(df):
    c = next((c for c in df.columns if c.lower() == "source"), None)
    return df[c].values if c else np.array(["baseline"] * len(df))


# ----------------------------------------------------------------------
# Fold construction (grouped, leakage-free augmentation done once per fold)
# ----------------------------------------------------------------------
def _prep_pair(tr_df, va_df, dataset_type, syn_cfg, factor, augment):
    # Fit preprocessing (and, if requested, GMM augmentation) on the training
    # partition only, then transform both partitions.
    if augment:
        target = max(len(tr_df), int(round(len(tr_df) * factor)))
        tr_df, _ = augment_dataframe(tr_df, target, syn_cfg)
    pre = DataPreprocessor(dataset_type).fit(tr_df)
    X_tr, y_tr = pre.transform(tr_df)
    X_va, y_va = pre.transform(va_df)
    return X_tr, y_tr, X_va, y_va


def build_grouped_folds(df, groups, k, dataset_type, config, factor, augment):
    # Precompute, ONCE, the transformed arrays for every grouped CV fold so the
    # (expensive) augmentation is not recomputed for each model/config.
    syn_cfg = config.get("synthetic", {})
    df = df.reset_index(drop=True)
    groups = np.asarray(groups)
    k = min(k, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=k)
    folds = []
    for tr_idx, va_idx in gkf.split(df, groups=groups):
        tr = df.iloc[tr_idx].reset_index(drop=True)
        va = df.iloc[va_idx].reset_index(drop=True)
        folds.append(_prep_pair(tr, va, dataset_type, syn_cfg, factor, augment))
    return folds


def build_external(train_df, test_df, dataset_type, config, factor, augment):
    # Fit on the (optionally augmented) outer-train pool; transform the outer-test.
    syn_cfg = config.get("synthetic", {})
    train_df = train_df.reset_index(drop=True)
    if augment:
        target = max(len(train_df), int(round(len(train_df) * factor)))
        tr_aug, _ = augment_dataframe(train_df, target, syn_cfg)
    else:
        tr_aug = train_df
    pre = DataPreprocessor(dataset_type).fit(tr_aug)
    X_tr, y_tr = pre.transform(tr_aug)
    X_te, y_te = pre.transform(test_df)
    return X_tr, y_tr, X_te, y_te, pre


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def cv_score(folds, model_name, params, config) -> Dict[str, float]:
    # Mean/std MAPE and R2 of a model over precomputed folds (validation = real).
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
    return {"mean_mape": float(np.mean(mapes)), "std_mape": float(np.std(mapes)),
            "mean_r2": float(np.mean(r2s)), "std_r2": float(np.std(r2s))}


def external_score(ext, model_name, params, config):
    # Train on the (aug or base) outer-train pool, evaluate on the outer-test set.
    rs = config.get("evaluation", {}).get("random_state", 42)
    ann_def = config["training"].get("ann", {})
    kan_def = config["training"].get("kan", {})
    X_tr, y_tr, X_te, y_te, pre = ext
    model = make_model(model_name, params, random_state=rs, ann_defaults=ann_def, kan_defaults=kan_def)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mape, _, r2 = _metrics(y_te, pred)
    return mape, r2, model, pre, pred


# ----------------------------------------------------------------------
# Selection CV (single-level grouped) — feeds the appendix and best configs
# ----------------------------------------------------------------------
def selection_cv(baseline, groups, dataset_type, config, factor, grids):
    # Grouped CV over the full real baseline for every configuration. Used only
    # for hyperparameter selection/reporting (appendix), never as a test metric.
    k = config.get("evaluation", {}).get("k_outer", 5)
    print("  Selection CV: precomputing grouped folds (augmented)...")
    sel_aug = build_grouped_folds(baseline, groups, k, dataset_type, config, factor, augment=True)
    sel_base = build_grouped_folds(baseline, groups, k, dataset_type, config, factor, augment=False)

    all_rows: List[Dict[str, Any]] = []
    for model_name, param_list in grids.items():
        t0 = time.time()
        for params in param_list:
            try:
                res = cv_score(sel_aug, model_name, params, config)
                all_rows.append({"model": model_name, "params": json.dumps(params), **res})
            except Exception as e:
                print(f"    [{model_name}] ERROR: {e}")
        print(f"    [{model_name}] {len(param_list)} configs ({time.time() - t0:.1f}s)")

    all_df = pd.DataFrame(all_rows).sort_values("mean_mape").reset_index(drop=True)
    return all_df, sel_aug, sel_base


# ----------------------------------------------------------------------
# Synthetic-sample-count sensitivity (reviewer comment D)
# ----------------------------------------------------------------------
def aug_sensitivity(baseline, groups, dataset_type, config, model_name, params):
    # Grouped selection-CV MAPE of the chosen model as the number of synthetic
    # samples per training pool is varied (including the no-synthetic baseline).
    k = config.get("evaluation", {}).get("k_outer", 5)
    n_base = len(baseline)
    targets = config.get("synthetic", {}).get("sensitivity_targets", [150, 300, 600])
    # Always include the no-synthetic baseline, then the augmented targets above it.
    targets = sorted({n_base} | {t for t in targets if t > n_base})
    rows = []
    for target in targets:
        factor = target / max(1, n_base)
        augment = target > n_base
        folds = build_grouped_folds(baseline, groups, k, dataset_type, config, factor, augment)
        res = cv_score(folds, model_name, params, config)
        rows.append({"target_samples": int(target), "augment": augment,
                     "cv_mape": res["mean_mape"], "cv_mape_std": res["std_mape"],
                     "cv_r2": res["mean_r2"]})
        print(f"    target={target:4d} (aug={augment}): MAPE={res['mean_mape']:.2f}% +/- {res['std_mape']:.2f}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Main per-dataset driver
# ----------------------------------------------------------------------
def run_dataset(dataset_type: str, config: Dict[str, Any]) -> None:
    output_dir = Path(config["data"]["output_dir"])
    baseline = pd.read_csv(output_dir / f"{dataset_type}_baseline.csv")
    groups = config_group_labels(baseline)
    n_groups = len(np.unique(groups))

    eval_cfg = config.get("evaluation", {})
    k_outer = min(eval_cfg.get("k_outer", 5), n_groups)
    k_inner = eval_cfg.get("k_inner", 3)
    target_samples = config.get("synthetic", {}).get("target_samples", 300)
    factor = target_samples / max(1, len(baseline))

    print(f"\n{'=' * 64}\nDataset: {dataset_type}  |  real={len(baseline)}  configs(groups)={n_groups}\n"
          f"  nested grouped CV: outer={k_outer}, inner={k_inner}  |  aug factor={factor:.2f}x\n{'=' * 64}")

    grids = get_param_grids(config["training"])

    # ---- 1) Selection CV (appendix, best config per model) ----
    all_df, _sel_aug, _sel_base = selection_cv(baseline, groups, dataset_type, config, factor, grids)
    all_df.to_excel(output_dir / f"{dataset_type}_all_results.xlsx", index=False)
    ann_df = all_df[all_df["model"] == "ANN"].copy().reset_index(drop=True)
    if not ann_df.empty:
        ann_df.insert(0, "rank", np.arange(1, len(ann_df) + 1))
        ann_df.to_excel(output_dir / f"{dataset_type}_ann_grid.xlsx", index=False)

    best_configs = {}
    sel_best = {}
    for model_name in grids:
        sub = all_df[all_df["model"] == model_name]
        if sub.empty:
            continue
        best = sub.iloc[0]
        best_configs[model_name] = json.loads(best["params"])
        sel_best[model_name] = {"cv_aug_mape": best["mean_mape"], "cv_aug_mape_std": best["std_mape"],
                                "cv_aug_r2": best["mean_r2"]}

    # ---- 2) Nested grouped CV (unbiased), augmented vs baseline (paired) ----
    print(f"\n  Nested grouped CV ({k_outer} outer x {k_inner} inner folds)...")
    gkf = GroupKFold(n_splits=k_outer)
    families = list(grids.keys())
    fold_scores = {m: {"aug_mape": [], "aug_r2": [], "base_mape": [], "base_r2": []} for m in families}
    oof = {m: np.full(len(baseline), np.nan) for m in families}
    oof_base = {m: np.full(len(baseline), np.nan) for m in families}

    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(baseline, groups=groups), start=1):
        t0 = time.time()
        otr = baseline.iloc[tr_idx].reset_index(drop=True)
        ote = baseline.iloc[te_idx].reset_index(drop=True)
        otr_groups = groups[tr_idx]

        # Inner selection folds (augmented) + outer-test transforms, built once.
        inner_aug = build_grouped_folds(otr, otr_groups, k_inner, dataset_type, config, factor, augment=True)
        ext_aug = build_external(otr, ote, dataset_type, config, factor, augment=True)
        ext_base = build_external(otr, ote, dataset_type, config, factor, augment=False)

        for model_name in families:
            # Inner grid search: pick this family's hyperparameters on inner folds.
            best_params, best_mape = None, float("inf")
            for params in grids[model_name]:
                try:
                    m = cv_score(inner_aug, model_name, params, config)["mean_mape"]
                except Exception:
                    continue
                if m < best_mape:
                    best_mape, best_params = m, params
            if best_params is None:
                continue

            try:
                mape_a, r2_a, _, _, pred_a = external_score(ext_aug, model_name, best_params, config)
                mape_b, r2_b, _, _, pred_b = external_score(ext_base, model_name, best_params, config)
            except Exception as e:
                print(f"      [{model_name}] outer-eval error: {e}")
                continue
            fold_scores[model_name]["aug_mape"].append(mape_a)
            fold_scores[model_name]["aug_r2"].append(r2_a)
            fold_scores[model_name]["base_mape"].append(mape_b)
            fold_scores[model_name]["base_r2"].append(r2_b)
            oof[model_name][te_idx] = pred_a
            oof_base[model_name][te_idx] = pred_b
        print(f"    outer fold {fold_i}/{k_outer}: train={len(otr)} test={len(ote)} ({time.time() - t0:.1f}s)")

    # ---- Aggregate nested results (paired aug vs base) ----
    comparison = []
    for model_name in families:
        s = fold_scores[model_name]
        if not s["aug_mape"]:
            continue
        aug_mape = np.array(s["aug_mape"]); base_mape = np.array(s["base_mape"])
        # Paired test across outer folds: does augmentation change the error?
        try:
            _, pval = stats.ttest_rel(base_mape, aug_mape)
        except Exception:
            pval = float("nan")
        comparison.append({
            "model": model_name,
            "params": json.dumps(best_configs.get(model_name, {})),
            "cv_aug_mape": sel_best.get(model_name, {}).get("cv_aug_mape", float("nan")),
            "cv_aug_mape_std": sel_best.get(model_name, {}).get("cv_aug_mape_std", float("nan")),
            "cv_aug_r2": sel_best.get(model_name, {}).get("cv_aug_r2", float("nan")),
            # Nested (unbiased) estimate, augmented:
            "test_mape": float(np.mean(aug_mape)), "test_mape_std": float(np.std(aug_mape)),
            "test_r2": float(np.mean(s["aug_r2"])), "test_r2_std": float(np.std(s["aug_r2"])),
            # Nested (unbiased) estimate, baseline-only (no GMM synthetic):
            "test_mape_base": float(np.mean(base_mape)), "test_mape_base_std": float(np.std(base_mape)),
            "test_r2_base": float(np.mean(s["base_r2"])),
            # Augmentation effect (positive delta = augmentation increases error):
            "aug_delta_mape": float(np.mean(aug_mape) - np.mean(base_mape)),
            "aug_paired_p": float(pval),
        })

    comp_df = pd.DataFrame(comparison).sort_values("test_mape").reset_index(drop=True)
    comp_df.to_excel(output_dir / f"{dataset_type}_model_comparison.xlsx", index=False)

    # ---- Out-of-fold predictions (all real samples predicted while held out) ----
    test_predictions = {"y_true": baseline[_target_col(baseline)].values, "source": _source_vals(baseline)}
    for model_name in families:
        test_predictions[f"pred_{model_name}"] = oof[model_name]       # +GMM augmented OOF
        test_predictions[f"pred_base_{model_name}"] = oof_base[model_name]  # real-data-only OOF
    pd.DataFrame(test_predictions).to_csv(output_dir / f"{dataset_type}_test_predictions.csv", index=False)
    with open(output_dir / f"{dataset_type}_best_configs.json", "w") as f:
        json.dump(best_configs, f, indent=2)

    # ---- Deployable ANN artifact: retrain best ANN on full augmented baseline ----
    if "ANN" in best_configs:
        ext_full = build_external(baseline, baseline, dataset_type, config, factor, augment=True)
        _, _, model, pre, _ = external_score(ext_full, "ANN", best_configs["ANN"], config)
        import torch
        torch.save({"model_state_dict": model.model.state_dict(),
                    "config": model.get_config(), "input_dim": model.input_dim},
                   output_dir / f"{dataset_type}_final_ann.pt")
        with open(output_dir / f"{dataset_type}_final_preprocessor.pkl", "wb") as f:
            pickle.dump(pre.__dict__, f)

    # ---- 3) Synthetic-sample-count sensitivity for the best model ----
    best_overall = comp_df.iloc[0]["model"]
    print(f"\n  Synthetic-sample-count sensitivity (best model: {best_overall})...")
    sens = aug_sensitivity(baseline, groups, dataset_type, config, best_overall, best_configs[best_overall])
    sens.to_csv(output_dir / f"{dataset_type}_aug_sensitivity.csv", index=False)

    # ---- Report ----
    print(f"\n  === Nested grouped CV ({dataset_type}) — sorted by nested test MAPE ===")
    for _, r in comp_df.iterrows():
        star = "*" if r["aug_paired_p"] < 0.05 else " "
        print(f"    {r['model']:16s} test={r['test_mape']:5.2f}+/-{r['test_mape_std']:.2f}%  "
              f"R2={r['test_r2']:.3f} | base={r['test_mape_base']:5.2f}%  "
              f"d(aug)={r['aug_delta_mape']:+.2f}%{star}")


def main():
    parser = argparse.ArgumentParser(description="Nested grouped cross-validation training and model comparison")
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
